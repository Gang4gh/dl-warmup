"""Train and evaluate deep title generation model.
refered projects:
    [] https://github.com/tensorflow/models/tree/master/official/transformer/v2
    [] https://github.com/google/trax/tree/master/trax
    [] https://github.com/cerebroai/reformers *
    [] https://github.com/lucidrains/reformer-pytorch/tree/master/reformer_pytorch
*: this project is incomplete, don't recommend to follow
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
    params["val_data_dir"] = flags_obj.val_data_dir
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
    params["num_hashes"] = flags_obj.num_hashes
    params["test_num_hashes"] = flags_obj.test_num_hashes
    params["use_full_attention_in_reformer"] = flags_obj.use_full_attention_in_reformer
    params["bucket_size"] = flags_obj.bucket_size

    if flags_obj.one_dropout is not None:
      params['layer_postprocess_dropout'] = flags_obj.one_dropout
      params['attention_dropout'] = flags_obj.one_dropout
      params['relu_dropout'] = flags_obj.one_dropout
    if flags_obj.attention_dropout is not None:
      params['attention_dropout'] = flags_obj.attention_dropout
    params['lsh_attention_dropout'] = params['attention_dropout'] if params["use_full_attention_in_reformer"] else flags_obj.lsh_attention_dropout
    logging.info(f'dropouts (postprocess, attention, lsh_attention, relu) = {[params[k] for k in ["layer_postprocess_dropout", "attention_dropout", "lsh_attention_dropout", "relu_dropout"]]}')
    logging.info(f'attention_padding_strategy = {flags_obj.attention_padding_strategy}')

    assert self.flags_obj.vocab_file, 'vocab file is None'
    self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(self.flags_obj.vocab_file)
    self.EOS_id = self.tokenizer.encode('<EOS>')[0]
    params["vocab_size"] = self.tokenizer.vocab_size
    logging.info('loaded vocab from {}, vocab_size={} and EOS_id={}'.format(self.flags_obj.vocab_file, self.tokenizer.vocab_size, self.EOS_id))

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


  def create_model(self, mode):
    logging.info('use_reformer = {}'.format(self.flags_obj.use_reformer))
    if self.flags_obj.use_reformer:
      logging.info(f'num_hashes, test_num_hashes = {self.params["num_hashes"]}, {self.params["test_num_hashes"]}')
      logging.info(f'allow_duplicated_attention = {self.flags_obj.allow_duplicated_attention}')
      return reformer.create_model(self.params, mode=mode)
    else:
      return transformer.create_model(self.params, mode=mode)


  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(
        enable_xla=flags_obj.enable_xla)

    train_ds = self._create_dataset(params['data_dir'], repeat=None)
    val_ds = self._create_dataset(params['val_data_dir'] or re.sub(r'-training.*', '-test.dtitle.gz', params['data_dir']), repeat=1)
    val_ds = val_ds.take(flags_obj.validation_example_count // params["batch_size"]).cache()

    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = self.create_model(mode='train')
      model.compile(optimizer=self._create_optimizer(params), loss=self._create_loss_fn(params))

    if not os.path.exists(flags_obj.model_dir):
      os.mkdir(flags_obj.model_dir)

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_mgr = tf.train.CheckpointManager(checkpoint, flags_obj.model_dir, max_to_keep=3, keep_checkpoint_every_n_hours=12)
    if ckpt_mgr.latest_checkpoint:
      #self._print_variables_and_exit(flags_obj.model_dir)
      model.fit([tf.ones([params["batch_size"], params['max_input_length']], tf.int32), tf.ones([params["batch_size"], params['max_target_length']], tf.int32)],
          tf.ones([params["batch_size"], params['max_target_length']], tf.int32),
          verbose=0)
      checkpoint.restore(ckpt_mgr.latest_checkpoint).assert_consumed()
      current_step = model.optimizer.iterations.numpy() - 1
      logging.info("Loaded checkpoint %s, current_step %d", ckpt_mgr.latest_checkpoint, current_step)

    if current_step >= flags_obj.train_steps:
      logging.info("Reach the target train_steps({}) and exit.".format(flags_obj.train_steps))
      return None

    logging.info(f'Start train iteration at global step: {current_step}')
    model.summary()
    #print(model.variables)
    history = model.fit(
        train_ds,
        initial_epoch=current_step // flags_obj.steps_between_evals,
        epochs=(flags_obj.train_steps-1) // flags_obj.steps_between_evals + 1,
        steps_per_epoch=min(flags_obj.steps_between_evals, flags_obj.train_steps - current_step),
        callbacks=self._create_callbacks(flags_obj.model_dir, current_step, params, ckpt_mgr),
        validation_data=val_ds,
        validation_steps=flags_obj.validation_example_count // params["batch_size"], # redundant but suppress one warining
        verbose=1)
    logging.info("Train history: {}".format(history.history))
    current_step = model.optimizer.iterations.numpy() - 1
    logging.info("End train iteration at global step:{}".format(current_step))

    return history

  def eval(self):
    """Evaluates the model."""
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = self.create_model(mode='eval')
      model.compile(loss=self._create_loss_fn(self.params))
      model.summary()
      self._load_model_weights(model)

    N = self.flags_obj.validation_example_count // self.params["batch_size"]
    ds = self._create_dataset(self.params['data_dir'], repeat=1).take(N)
    res = model.evaluate(ds, steps=N)
    logging.info('Evaluate {} batches, res={}'.format(N, res))

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

    model = self.create_model(mode='predict')
    model.summary()
    self._load_model_weights(model)

    #numpy.set_printoptions(threshold=sys.maxsize)

    ds = self._create_dataset(params['data_dir'], repeat=1, batch_size=1)
    logging.info('max prediction limit = {}'.format(flags_obj.max_predict_count))
    if flags_obj.max_predict_count:
      ds = ds.take(flags_obj.max_predict_count)

    inputs, input_strings, targets, target_strings, preds, pred_strings, pred_scores, null_probs = [], [], [], [], [], [], [], []
    for ((inp, tar), _) in ds.unbatch():
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
          f.write('{}\t{}\t{}\n'.format(inp[0], re.sub(r'[\t\r\n]+', ' ', html.unescape(pred)), null_prob)) # pred may contains '\n' after unescape
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
    assert checkpoint_path, 'Latest checkpoint does not exist or is invalid.'
    """Loads model weights when it is provided."""
    checkpoint.restore(checkpoint_path).expect_partial()
    logging.info("load model weights from: {}".format(checkpoint_path))

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

  def _create_random_dataset(self, vocab_size, batch_size, max_input_length, max_target_length):
    def _random_example_generator():
      while True:
        X = np.random.randint(1, vocab_size, (batch_size, max_input_length))
        Y = np.random.randint(1, vocab_size, (batch_size, max_target_length))
        yield X, Y

    ds = tf.data.Dataset.from_generator(_random_example_generator,
                                        output_types=(tf.int32, tf.int32),
                                        output_shapes=((batch_size, max_input_length), (batch_size, max_target_length)))
    return ds

  def _create_dtitle_dataset(self, data_file, batch_size, max_input_length, max_target_length, url_segment_limit, hostname_segment_limit, html_segment_limit, eos):
    def _dtitle_encode(ln):
      url, tar, hostname, html = tf.strings.split(ln, '\t')

      url = self.tokenizer.encode(url.numpy())
      hostname = self.tokenizer.encode(hostname.numpy())
      html = self.tokenizer.encode(html.numpy())
      tar = self.tokenizer.encode(tar.numpy())

      if self.flags_obj.input_concat_schema == 'v0':
        # baseline
        return html[:max_input_length - 1] + [eos], tar + [eos]
      elif self.flags_obj.input_concat_schema == 'v1':
        # concatenated
        url = [eos+1] + url[:url_segment_limit-2] + [eos]
        hostname = [eos+2] + hostname[:hostname_segment_limit-2] + [eos]
        html = [eos+3] + html[:html_segment_limit-2] + [eos]
        return url + hostname + html, tar + [eos]
      elif self.flags_obj.input_concat_schema == 'v2':
        # concatenated + fixed positins (padding)
        url = [eos+1] + url[:url_segment_limit - 2] + [eos] + [0] * max(0, url_segment_limit - 2 - len(url))
        hostname = [eos+2] + hostname[:hostname_segment_limit - 2] + [eos] + [0] * max(0, hostname_segment_limit - 2 - len(hostname))
        html = [eos+3] + html[:html_segment_limit-2] + [eos]
        return url + hostname + html, tar + [eos]
      elif self.flags_obj.input_concat_schema == 'v3':
        # fixed positins (padding)
        url = url[:url_segment_limit - 1] + [eos] + [0] * max(0, url_segment_limit - 1 - len(url))
        hostname = hostname[:hostname_segment_limit - 1] + [eos] + [0] * max(0, hostname_segment_limit - 1 - len(hostname))
        html = html[:html_segment_limit-1] + [eos]
        return url + hostname + html, tar + [eos]
      else:
        raise ValueError('invalid input_concat_schema: ' + self.flags_obj.input_concat_schema)

    ds = tf.data.TextLineDataset(data_file, compression_type='GZIP' if data_file.endswith('.gz') else None)
    ds = ds.map(lambda ln: tf.py_function(_dtitle_encode, [ln], [tf.int32, tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.filter(lambda _, target: tf.size(target) <= max_target_length)
    ds = ds.padded_batch(batch_size, padded_shapes=([max_input_length], [max_target_length]), drop_remainder=True)
    return ds

  def _create_tfrecord_dataset(self, data_file, batch_size, max_input_length, max_target_length):
    def _convert_proto_to_tensor(proto):
      X = tf.reshape(tf.io.parse_tensor(proto, tf.int32), shape=[-1, max_input_length + max_target_length])
      return X[:, :max_input_length], X[:, max_input_length:]

    ds = tf.data.TFRecordDataset(data_file, compression_type='GZIP' if data_file.endswith('.gz') else None)
    ds = ds.map(_convert_proto_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.unbatch().batch(batch_size, drop_remainder=True)
    return ds

  def _create_tokenized_tfrecord_dataset(self, data_file, batch_size, max_input_length, max_target_length, url_segment_limit, hostname_segment_limit, html_segment_limit, eos):
    ds = tf.data.TFRecordDataset(data_file, compression_type='GZIP' if data_file.endswith('.gz') else None)
    description = {
      'url': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'title': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'hostname': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'html': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      }
    def _tf_parse_and_truncate(proto):
      ex = tf.io.parse_single_example(proto, description)
      return [ tf.concat([[eos+1], tf.cast(ex['url'][:url_segment_limit-2], tf.int32), [eos]], axis=0),
             tf.concat([[eos+2], tf.cast(ex['hostname'][:hostname_segment_limit-2], tf.int32), [eos]], axis=0),
             tf.concat([[eos+3], tf.cast(ex['html'][:html_segment_limit-2], tf.int32), [eos]], axis=0),
             tf.concat([tf.cast(ex['title'], tf.int32), [eos]], axis=0) ]
    ds = ds.map(_tf_parse_and_truncate, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.filter(lambda _a, _b, _c, target: tf.size(target) <= max_target_length)
    ds = ds.padded_batch(batch_size, padded_shapes=([url_segment_limit], [hostname_segment_limit], [html_segment_limit], [max_target_length]), drop_remainder=True)
    ds = ds.map(lambda url, hostname, html, title: (tf.concat([url, hostname, html], axis=-1), title))
    return ds

  def _create_dataset(self, data_file, repeat, batch_size=None, shuffle_size=None, create_cache=False):
    batch_size = batch_size or self.params['batch_size']
    max_input_length = self.params['max_input_length']
    max_target_length = self.params['max_target_length']
    url_segment_limit = 64 # max url length
    hostname_segment_limit = 64 # max hostname length
    html_segment_limit = max_input_length - url_segment_limit - hostname_segment_limit # max html length

    if data_file == '__random_input__':
      logging.info(f'open one random dataset.')
      ds = self._create_random_dataset(self.params["vocab_size"], batch_size, max_input_length, max_target_length)
    elif data_file.endswith('.dtitle') or data_file.endswith('.dtitle.gz'):
      logging.info(f'open one dtitle dataset from "{data_file}".')
      ds = self._create_dtitle_dataset(data_file, batch_size, max_input_length, max_target_length, url_segment_limit, hostname_segment_limit, html_segment_limit, self.EOS_id)
    elif data_file.endswith('.tfrecord') or data_file.endswith('.tfrecord.gz'):
      logging.info(f'open one tfrecord dataset from "{data_file}".')
      ds = self._create_tfrecord_dataset(data_file, batch_size, max_input_length, max_target_length)
    elif data_file.endswith('.tokenized-tfrecord') or data_file.endswith('tokenized-tfrecord.gz'):
      logging.info(f'open one tokenized-tfrecord dataset from "{data_file}".')
      ds = self._create_tokenized_tfrecord_dataset(data_file, batch_size, max_input_length, max_target_length, url_segment_limit, hostname_segment_limit, html_segment_limit, self.EOS_id)
    else:
      raise ValueError(f'invalid input file format: {data_file}')

    cache_desc = f'{data_file}.{batch_size}_{max_input_length}_{max_target_length}_{url_segment_limit}_{hostname_segment_limit}.cache'
    if create_cache or os.path.isfile(f'{cache_desc}.index'):
      ds = ds.cache(cache_desc)
    if repeat != 1:
      ds = ds.repeat(repeat)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size // batch_size)
    ds = ds.map(lambda x, y: ((x, y), y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

  def _print_variables_and_exit(self, checkpoint_dir):
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    for var in tf.train.list_variables(ckpt_path):
      print(var)
    sys.exit()

  def convert_dtitle_to_tfrecord(self, max_batch_count=None, logging_step=1, compression_type='GZIP'):
    data_file = self.params['data_dir']
    if data_file.endswith('.dtitle'):
      tfrecord_file = data_file[:-7] + '.tfrecord'
    elif data_file.endswith('.dtitle.gz'):
      tfrecord_file = data_file[:-10] + '.tfrecord'
    else:
      raise ValueError(f'invalid data_file postfix : {data_file}')
    if compression_type == 'GZIP':
      tfrecord_file += '.gz'

    ds = self._create_dataset(data_file, repeat=1)
    with tf.io.TFRecordWriter(tfrecord_file, compression_type) as tfwriter:
      for batch, ((inp, tar), _) in enumerate(ds):
        if max_batch_count and batch == max_batch_count: break
        if (batch + 1) % (logging_step * 1024) == 0:
          logging.info(f'convert {(batch+1)//logging_step//1024}K batches')
        proto = tf.io.serialize_tensor(tf.concat([inp, tar], axis=-1)).numpy()
        tfwriter.write(proto)
    logging.info(f'convert {batch} batches in total.')

def test_read_and_dump_datasets(task):
  def _dump_to_file(filename, data_file, batch_count, decode_fn, repeat=1):
    logging.info(f'read data_file {data_file}, batch_count={batch_count}')
    ds = task._create_dataset(data_file, repeat=1)
    for r in range(repeat):
      start_time = time.time()
      out_filename = filename + f'-r{r}.log'
      logging.info(f'round {r}, write output to {out_filename}')
      with open(out_filename, 'w') as ow:
        for batch, ((inp, tar), _) in enumerate(ds):
          if batch == batch_count: break
          for (index, (inp1, tar1)) in enumerate(zip(inp, tar)):
            htmlbody = decode_fn(inp1.numpy(), 3, concatenate_segments=False)
            title = decode_fn(tar1.numpy(), concatenate_segments=False)
            ow.write('{}.{}:\ninp = {}\ntar = {}\ninp_str = {}\ntar_str = {}'.format(batch, index, inp1, tar1, htmlbody, title))
      logging.info(f'end of round {r}, --- {int(time.time() - start_time)} seconds ---')

  np.set_printoptions(threshold=2048)
  repeat_times = 1
  for idx, bc in enumerate([100, 1000, 10000]):
    #_dump_to_file(f'out-random-{idx}.batch{bc}', '__random_input__', batch_count=bc, decode_fn=task._trim_and_decode, repeat=repeat_times)
    _dump_to_file(f'out-dtitle-gz-{idx}.batch{bc}', task.params['data_dir'] + '.gz', batch_count=bc, decode_fn=task._trim_and_decode, repeat=repeat_times)
    _dump_to_file(f'out-tfrecord-gz-{idx}.batch{bc}', task.params['data_dir'].replace('.dtitle', '.tfrecord.gz'), batch_count=bc, decode_fn=task._trim_and_decode, repeat=repeat_times)
    _dump_to_file(f'out-tokenized-tfrecord-gz-{idx}.batch{bc}', task.params['data_dir'].replace('.dtitle', '.tokenized-tfrecord.gz'), batch_count=bc, decode_fn=task._trim_and_decode, repeat=repeat_times)


def test(task):
  test_read_and_dump_datasets(task)

def main(_):
  flags_obj = flags.FLAGS
  task = Seq2SeqTask(flags_obj)
  if flags_obj.mode == "train":
    task.train()
  elif flags_obj.mode == "train-cache":
    ds = task._create_dataset(task.params['data_dir'], repeat=1, create_cache=True)
    for idx, ((inp, tar), _) in enumerate(ds):
      if idx % 1024 == 0:
        print(f'read {idx//1024}K batches')
      pass
  elif flags_obj.mode == "train-prep":
    # build-in dataset.cache generates 2+ uncompressed files, which might not fit in some scenarios
    task.convert_dtitle_to_tfrecord()
  elif flags_obj.mode == "predict":
    task.predict()
  elif flags_obj.mode == "eval":
    task.eval()
  elif flags_obj.mode == 'test':
    test(task)
  else:
    raise ValueError(f'invalid mode : {flags_obj.mode}')


if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
  misc.define_transformer_flags()
  app.run(main)
