"""Train and evaluate deep title generation model.
Derived from:
    https://github.com/tensorflow/models/tree/master/official/transformer/v2
    https://github.com/cerebroai/reformers
"""

import os
import sys
import time
import re
import numpy as np
import html

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds

import misc
from official.transformer.v2 import optimizer
import transformer
import reformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.utils.misc import distribution_utils
import metrics

from data_dtitle.process_dtitle_data import dtitle_reader

class Seq2SeqTask(object):
  """Main entry of Seq2Seq model."""

  def __init__(self, flags_obj):
    """Init function

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.
    """
    self.flags_obj = flags_obj

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_input_length"] = flags_obj.max_input_length
    params["max_target_length"] = flags_obj.max_target_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["num_parallel_calls"] = (flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size * max(num_gpus, 1)
    logging.info('actual batch_size = {} * {}'.format(flags_obj.batch_size, max(num_gpus, 1)))
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training

    self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(self.flags_obj.vocab_file)
    self.EOS_id = self.tokenizer.encode('<EOS>')[0]
    params["vocab_size"] = self.tokenizer.vocab_size
    logging.info('loaded vocab from {}, vocab_size={} and EOS_id={}'.format(self.flags_obj.vocab_file, self.tokenizer.vocab_size, self.EOS_id))

    logging.info('use input schema: {}'.format(self.flags_obj.input_schema))

    if params["dtype"] == tf.float16:
      # TODO(reedwm): It's pretty ugly to set the global policy in a constructor
      # like this. What if multiple instances of Seq2SeqTask are created?
      # We should have a better way in the tf.keras.mixed_precision API of doing
      # this.
      loss_scale = flags_core.get_loss_scale(flags_obj,
                                             default_for_fp16="dynamic")
      policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
          "mixed_float16", loss_scale=loss_scale)
      tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

    self.distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=num_gpus,
        tpu_address=flags_obj.tpu or "")
    logging.info("Running dtitle model with num_gpus = %d", num_gpus)

    if self.distribution_strategy:
      logging.info("For training, using distribution strategy: %s",
                   self.distribution_strategy)
    else:
      logging.info("Not using any distribution strategy.")


  def create_model(self, is_train):
    logging.info('use_reformer = {}'.format(self.flags_obj.use_reformer))
    if self.flags_obj.use_reformer:
      return reformer.create_model(self.params, is_train=is_train)
    else:
      return transformer.create_model(self.params, is_train=is_train)


  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(
        enable_xla=flags_obj.enable_xla)

    train_ds = self._create_dataset(params['data_dir'], repeat=None)
    test_ds = self._create_dataset(params['data_dir'].replace('training', 'test'), repeat=1)

    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = self.create_model(is_train=True)
      model.compile(optimizer=self._create_optimizer(params), loss=self._create_loss_fn(params))

    if not os.path.exists(flags_obj.model_dir):
      os.mkdir(flags_obj.model_dir)

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_mgr = tf.train.CheckpointManager(checkpoint, flags_obj.model_dir, max_to_keep=3, keep_checkpoint_every_n_hours=12)
    if ckpt_mgr.latest_checkpoint:
      #self._print_variables_and_exit(flags_obj.model_dir)
      model.fit((np.ones((1, params['max_input_length']), np.int32), np.ones((1, params['max_target_length']), np.int32)),
          np.ones((1, params['max_target_length']), np.int32),
          verbose=0)
      checkpoint.restore(ckpt_mgr.latest_checkpoint).assert_consumed()
      current_step = model.optimizer.iterations.numpy() - 1
      logging.info("Loaded checkpoint %s, current_step %d", ckpt_mgr.latest_checkpoint, current_step)

    if current_step >= flags_obj.train_steps:
      logging.info("Reach the target train_steps({}) and exit.".format(flags_obj.train_steps))
      return None

    logging.info("Start train iteration at global step:{}".format(current_step))
    model.summary()
    #print(model.variables)
    history = model.fit(
        train_ds,
        initial_epoch=current_step // flags_obj.steps_between_evals,
        epochs=(flags_obj.train_steps-1) // flags_obj.steps_between_evals + 1,
        steps_per_epoch=min(flags_obj.steps_between_evals, flags_obj.train_steps - current_step),
        callbacks=self._create_callbacks(flags_obj.model_dir, current_step, params, ckpt_mgr),
        validation_data=test_ds,
        validation_steps=flags_obj.validation_example_count // params["batch_size"],
        verbose=1)
    logging.info("Train history: {}".format(history.history))
    current_step = model.optimizer.iterations.numpy() - 1
    logging.info("End train iteration at global step:{}".format(current_step))

    return history

  def eval(self):
    """Evaluates the model."""
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = self.create_model(is_train=True)
      model.compile(loss=self._create_loss_fn(self.params))
      model.summary()
      self._load_model_weights(model)

    N = 128
    ds = self._create_dataset(self.params['data_dir'], repeat=1)
    res = model.evaluate(ds, steps=N)
    logging.info('Evaluate {} steps, res={}'.format(N, res))

  _UNDERSCORE_REPLACEMENT = "\\&undsc"
  def _decode_and_fix(self, ids):
    return self.tokenizer.decode(ids).replace(self._UNDERSCORE_REPLACEMENT, '_')

  def _trim_and_decode(self, ids, segment_count = 1, concatenate_segments = True):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    ids = [id for id in list(ids) if id]
    indexes = [0]
    try:
      for _ in range(segment_count):
        indexes.append(indexes[-1] + ids[indexes[-1]:].index(self.EOS_id) + 1)
      if concatenate_segments:
        return self._decode_and_fix(ids[:indexes[-1]-1])
      else:
        return [self._decode_and_fix(ids[indexes[i]:indexes[i+1]-1]) for i in range(len(indexes)-1)]
    except ValueError:  # No enough EOS found in input
      return self._decode_and_fix(ids)


  def _calculate_rouge_scores(self, summaries, references, max_length=None):
    # command to install pythonrouge: pip install git+https://github.com/tagucci/pythonrouge.git
    from pythonrouge.pythonrouge import Pythonrouge

    logging.info('calculate ROUGE scores of %d summaries', len(summaries))
    rouge = Pythonrouge(summary_file_exist=False,
                          summary=summaries, reference=references,
                          n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                          recall_only=False, stemming=True, stopwords=False,
                          word_level=True, length_limit=max_length is not None, length=max_length,
                          use_cf=False, cf=95, scoring_formula='average',
                          resampling=True, samples=1000, favor=True, p=0.5)
    scores = rouge.calc_score()
    logging.info('ROUGE(1/2/L) Scores:')
    logging.info('>   ROUGE-1-R/F1: %f / %f', scores['ROUGE-1-R'], scores['ROUGE-1-F'])
    logging.info('>   ROUGE-2-R/F1: %f / %f', scores['ROUGE-2-R'], scores['ROUGE-2-F'])
    logging.info('>   ROUGE-L-R/F1: %f / %f', scores['ROUGE-L-R'], scores['ROUGE-L-F'])
    avg_token_count = sum(len(' '.join(summary).split()) for summary in summaries) / len(summaries)
    avg_token_count_ref = sum(len(' '.join(summary[0]).split()) for summary in references) / len(references)
    logging.info('>   averageToken: %f / %f', avg_token_count, avg_token_count_ref)
    return scores

  def predict(self):
    """Predicts result from the model."""
    params = self.params
    flags_obj = self.flags_obj

    model = self.create_model(is_train=False)
    model.summary()
    self._load_model_weights(model)

    #numpy.set_printoptions(threshold=sys.maxsize)

    ds = self._create_dataset(params['data_dir'], repeat=1, batch_size=1)
    logging.info('max prediction limit = {}'.format(flags_obj.max_predict_count))
    if flags_obj.max_predict_count:
      ds = ds.take(flags_obj.max_predict_count)

    inputs, input_strings, targets, target_strings, preds, pred_strings, pred_scores, null_probs = [], [], [], [], [], [], [], []
    for ((inp, tar), _) in ds:
      inputs.append(inp.numpy())
      input_strings.append([re.sub(r'^<BOS#\d>', '', s) for s in self._trim_and_decode(inputs[-1], 3, concatenate_segments=False)])
      targets.append(tar.numpy())
      target_strings.append([self._trim_and_decode(targets[-1])])
    logging.info('load {} examples from {}'.format(len(targets), params['data_dir']))
    X = np.vstack(inputs)
    Y = np.ones([len(inputs), 1], np.int32)

    correct, total = 0, 0
    mpred = model.predict([X, Y], batch_size=params['batch_size'])
    for ind, (pred_ids, score, logits) in enumerate(zip(*mpred)):
      preds.append(pred_ids)
      pred_scores.append(score)
      null_probs.append(tf.nn.softmax(logits[0])[1])
      pred_strings.append(self._trim_and_decode(preds[-1]))
      total += 1
      correct += 1 if pred_strings[-1] == target_strings[ind][0] else 0
    logging.info('Test accuracy: {}/{}={}'.format(correct, total, correct/total))

    scores = self._calculate_rouge_scores(pred_strings, target_strings) if flags_obj.calc_rouge_scores else None

    timestamp = int(time.time())
    if flags_obj.prediction_details_file:
      out_path = flags_obj.prediction_details_file
      if flags_obj.prediction_details_file == '#model_dir':
        out_path = os.path.join(self.flags_obj.model_dir, 'prediction-details-{}.txt'.format(timestamp))
      with open(out_path, 'w', encoding='utf8') as f:
        f.write('# Example Count = {}\n'.format(len(pred_strings)))
        f.write('# Accuracy = {}\n'.format(correct/total))
        f.write('# ROUGE scores = {}\n'.format(scores))
        ref_rows = {}
        if flags_obj.prediction_reference_file:
          ref_rows = {row.url:row for row in dtitle_reader(flags_obj.prediction_reference_file, 'cap_query,cap_url,cap_title,cap_snippet,url,hostname,visual_title,title,html')}
        for ind, (inp, tar, pred, score, null_prob) in enumerate(zip(input_strings, target_strings, pred_strings, pred_scores, null_probs)):
          row = ref_rows[inp[0]] if inp[0] in ref_rows else None
          cap_title_normalized = re.sub(r' +', ' ', re.sub(r'</?strong>', '', row.cap_title)).strip().lower() if row else None
          f.write('\n# [{}]\n'.format(ind))
          f.write('Url       = {}\n'.format(inp[0]))
          f.write('NullProb  = {}\n'.format(null_prob))
          f.write('Predict   = {}\n'.format(html.unescape(pred)))
          #f.write('PredScore = {}\n'.format(score))
          f.write('ProdTitle = {}\n'.format(cap_title_normalized))
          f.write('HtmlTitle = {}\n'.format(html.unescape(tar[0])))
          f.write('HostName  = {}\n'.format(inp[1]))
          f.write('Vis_Title = {}\n'.format(row.visual_title if row else None))
          f.write('Cap_Query = {}\n'.format(row.cap_query if row else None))
          f.write('Cap_Url   = {}\n'.format(row.cap_url if row else None))
          f.write('Cap_Title = {}\n'.format(row.cap_title if row else None))
          f.write('Cap_Snipt = {}\n'.format(row.cap_snippet if row else None))
          f.write('HtmlBody  = {}\n'.format(inp[2]))
      logging.info('write prediction details to {}'.format(out_path))

    if flags_obj.prediction_compact_file:
      out_path = flags_obj.prediction_compact_file
      if flags_obj.prediction_compact_file == '#model_dir':
        out_path = os.path.join(self.flags_obj.model_dir, 'prediction-compact-{}.txt'.format(timestamp))
      with open(out_path, 'w', encoding='utf8') as f:
        f.write('NormalizedUrl\tPredict\tNullProb\n')
        for inp, pred, null_prob in zip(input_strings, pred_strings, null_probs):
          f.write('{}\t{}\t{}\n'.format(inp[0], re.sub(r'[\t\n]+', ' ', html.unescape(pred)), null_prob)) # pred may contains '\n' after unescape
      logging.info('write compact prediction to {}'.format(out_path))


  def _create_callbacks(self, log_dir, init_steps, params, ckpt_mgr):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    callbacks = []
    callbacks.append(optimizer.LearningRateScheduler(sfunc, init_steps))
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: ckpt_mgr.save()))
    if self.flags_obj.enable_tensorboard:
      tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=log_dir, profile_batch=0, write_graph=False,
          update_freq=self.flags_obj.steps_between_tensorboard_log * params['batch_size'])
      callbacks.append(tensorboard_callback)
    callbacks.append(tf.keras.callbacks.CSVLogger('{}/history.step-{}.log'.format(log_dir, init_steps)))
    return callbacks

  def _load_model_weights(self, model):
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = tf.train.latest_checkpoint(self.flags_obj.model_dir)
    """Loads model weights when it is provided."""
    if checkpoint_path:
      checkpoint.restore(checkpoint_path).expect_partial()
      logging.info("load model weights from: {}".format(checkpoint_path))
    else:
      logging.info('no checkpoint found from: {}'.format(checkpoint_path))

  def _create_optimizer(self, params):
    """Creates optimizer."""
    return tf.keras.optimizers.Adam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

  def _create_loss_fn(self, params):
    logging.info('use loss_fn: %s', self.flags_obj.loss_fn)
    if self.flags_obj.loss_fn == 'smoothed_cross_entropy':
      label_smoothing = params["label_smoothing"]
      vocab_size = params["vocab_size"]
      def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [params['batch_size'], -1])
        y_pred = tf.reshape(y_pred, [params['batch_size'], -1, vocab_size])
        return metrics.transformer_loss(y_pred, y_true, label_smoothing, vocab_size)
      return loss
    else:
      return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def _create_random_dataset(self, dtitle_file=None, repeat=None, batch_size=None):
    vocab_size = self.params["vocab_size"]
    batch_size = batch_size or self.params['batch_size']
    max_input_length = self.params['max_input_length']
    max_target_length = self.params['max_target_length']

    def _random_example_generator():
      while True:
        X = np.random.randint(1, vocab_size, (batch_size, max_input_length))
        Y = np.random.randint(1, vocab_size, (batch_size, max_target_length))
        yield X, Y

    ds = tf.data.Dataset.from_generator(_random_example_generator,
                                        output_types=(tf.int32, tf.int32),
                                        output_shapes=((batch_size, max_input_length), (batch_size, max_target_length)))
    ds = ds.map(lambda x, y: ((x, y), y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def _create_dataset(self, dtitle_file, repeat, batch_size=None, shuffle_size=None):
    batch_size = batch_size or self.params['batch_size']
    max_input_length = self.params['max_input_length']
    max_target_length = self.params['max_target_length']
    user_segment_limit = 64 # max url segment length
    hostname_segment_limit = 64 # max hostname segment length

    eos = self.EOS_id

    def _data_encode(ln):
      url, tar, hostname, html = tf.strings.split(ln, '\t')

      url = self.tokenizer.encode(url.numpy())
      hostname = self.tokenizer.encode(hostname.numpy())
      html = self.tokenizer.encode(html.numpy())
      tar = self.tokenizer.encode(tar.numpy())

      if self.flags_obj.input_schema == 'v0':
        # baseline
        return html[:max_input_length - 1] + [eos], tar + [eos]
      elif self.flags_obj.input_schema == 'v1':
        # concatenated
        url = [eos+1] + url[:user_segment_limit-2] + [eos]
        hostname = [eos+2] + hostname[:hostname_segment_limit-2] + [eos]
        html = [eos+3] + html[:max_input_length-user_segment_limit-hostname_segment_limit-2] + [eos]
        return url + hostname + html, tar + [eos]
      elif self.flags_obj.input_schema == 'v2':
        # concatenated + fixed positins (padding)
        url = [eos+1] + url[:user_segment_limit - 2] + [eos] + [0] * max(0, user_segment_limit - 2 - len(url))
        hostname = [eos+2] + hostname[:hostname_segment_limit - 2] + [eos] + [0] * max(0, hostname_segment_limit - 2 - len(hostname))
        html = [eos+3] + html[:max_input_length - user_segment_limit - hostname_segment_limit - 2] + [eos]
        return url + hostname + html, tar + [eos]
      elif self.flags_obj.input_schema == 'v3':
        # fixed positins (padding)
        url = url[:user_segment_limit - 1] + [eos] + [0] * max(0, user_segment_limit - 1 - len(url))
        hostname = hostname[:hostname_segment_limit - 1] + [eos] + [0] * max(0, hostname_segment_limit - 1 - len(hostname))
        html = html[:max_input_length - user_segment_limit - hostname_segment_limit - 1] + [eos]
        return url + hostname + html, tar + [eos]
      else:
        raise ValueError('invalid input_schema:' + self.flags_obj.input_schema)

    ds = tf.data.TextLineDataset(dtitle_file)
    ds = ds.map(lambda ln: tf.py_function(_data_encode, (ln,), [tf.int32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.filter(lambda body, title: tf.size(title) <= max_target_length)
    ds = ds.repeat(repeat)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size)
    ds = ds.padded_batch(batch_size, padded_shapes=([max_input_length], [max_target_length]), drop_remainder=False)
    if batch_size and batch_size == 1:
      ds = ds.unbatch()
    ds = ds.map(lambda x, y: ((x, y), y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

  def _print_variables_and_exit(self, checkpoint_dir):
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    for var in tf.train.list_variables(ckpt_path):
      print(var)
    sys.exit()


def test(task):
  ds = task._create_dataset(task.params['data_dir'], repeat=1)
  N = 1
  logging.info('Begin read dataset, batch_count=%d, batch_size=%d', N, task.params["batch_size"])
  np.set_printoptions(threshold=2048)
  for batch, ((inp, tar), _) in enumerate(ds):
    if batch == N: break
    for (index, (inp1, tar1)) in enumerate(zip(inp, tar)):
      htmlbody = task._trim_and_decode(inp1.numpy(), 3)
      title = task._trim_and_decode(tar1.numpy())
      print('{}:\ninp = {}\ntar = {}\ninp_str = {}\ntar_str = {}'.format(batch*4+index, inp1, tar1, htmlbody, title))
      break
  logging.info('End of read')


def main(_):
  flags_obj = flags.FLAGS
  task = Seq2SeqTask(flags_obj)
  if flags_obj.mode == "train":
    task.train()
  elif flags_obj.mode == "predict":
    task.predict()
  elif flags_obj.mode == "eval":
    task.eval()
  elif flags_obj.mode == 'test':
    test(task)
  else:
    raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
  misc.define_transformer_flags()
  app.run(main)
