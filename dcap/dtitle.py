"""Train and evaluate deep title generation model.
Based on: https://github.com/tensorflow/models/tree/master/official/transformer/v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import numpy

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.transformer import compute_bleu
from official.transformer.v2 import data_pipeline
from official.transformer.v2 import metrics
import misc
from official.transformer.v2 import optimizer
import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import keras_utils
from official.utils.misc import distribution_utils


def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref,
                               distribution_strategy=None):
  """Translate file and report the cased and uncased bleu scores.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      params,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False,
      distribution_strategy=distribution_strategy)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file,
                          distribution_strategy=None):
  """Calculate and record the BLEU score.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  subtokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref, distribution_strategy)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["num_parallel_calls"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training

    self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(self.flags_obj.vocab_file)
    self.EOS_id = self.tokenizer.encode('<EOS>')[0]
    params["vocab_size"] = self.tokenizer.vocab_size
    print('loaded vocab from {}, vocab_size={} and EOS_id={}'.format(self.flags_obj.vocab_file, self.tokenizer.vocab_size, self.EOS_id))

    if params["dtype"] == tf.float16:
      # TODO(reedwm): It's pretty ugly to set the global policy in a constructor
      # like this. What if multiple instances of TransformerTask are created?
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
    logging.info("Running transformer with num_gpus = %d", num_gpus)

    if self.distribution_strategy:
      logging.info("For training, using distribution strategy: %s",
                   self.distribution_strategy)
    else:
      logging.info("Not using any distribution strategy.")

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy,
                        tf.distribute.experimental.TPUStrategy)
    return False

  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(
        enable_xla=flags_obj.enable_xla)

    self._ensure_dir(flags_obj.model_dir)
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()

      current_step = 0
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
      if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()#.assert_consumed() TODO: restore doesn't work actually
        current_step = opt.iterations.numpy()
        model.load_weights(latest_checkpoint).expect_partial()
        logging.info("Loaded checkpoint %s", latest_checkpoint)

      if params["use_ctl"]:
        train_loss_metric = tf.keras.metrics.Mean(
            "training_loss", dtype=tf.float32)
      else:
        model.compile(opt)

    model.summary()

    train_ds = self._create_dataset(params['data_dir'], batch_size=params["batch_size"], repeat=None)

    callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []
    while current_step < flags_obj.train_steps:
      remaining_steps = flags_obj.train_steps - current_step
      train_steps_per_eval = (
          remaining_steps if remaining_steps < flags_obj.steps_between_evals
          else flags_obj.steps_between_evals)
      current_iteration = current_step // flags_obj.steps_between_evals

      logging.info("Start train iteration at global step:{}".format(current_step))
      history = None

      history = model.fit(
          train_ds,
          initial_epoch=current_iteration,
          epochs=current_iteration + 1,
          steps_per_epoch=train_steps_per_eval,
          callbacks=callbacks,
          # If TimeHistory is enabled, progress bar would be messy. Increase
          # the verbose level to get rid of it.
          verbose=(2 if flags_obj.enable_time_history else 1))
      current_step += train_steps_per_eval
      logging.info("Train history: {}".format(history.history))

      logging.info("End train iteration at global step:{}".format(current_step))

      if (flags_obj.bleu_source and flags_obj.bleu_ref):
        uncased_score, cased_score = self.eval()
        cased_score_history.append([current_iteration + 1, cased_score])
        uncased_score_history.append([current_iteration + 1, uncased_score])

    stats = ({
        "loss": train_loss
    } if history is None else misc.build_stats(history, callbacks))
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history
    return stats

  def eval(self):
    """Evaluates the model."""
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      if not self.predict_model:
        self.predict_model = transformer.create_model(self.params, False)
      self._load_weights_if_possible(
          self.predict_model,
          tf.train.latest_checkpoint(self.flags_obj.model_dir))
      self.predict_model.summary()
    return evaluate_and_log_bleu(
        self.predict_model, self.params, self.flags_obj.bleu_source,
        self.flags_obj.bleu_ref, self.flags_obj.vocab_file,
        self.distribution_strategy if self.use_tpu else None)

  def _trim_and_decode(self, ids):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
      index = list(ids).index(self.EOS_id)
      return self.tokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
      return self.tokenizer.decode(ids)

  def predict(self):
    """Predicts result from the model."""
    params = self.params
    flags_obj = self.flags_obj

    with tf.name_scope("model"):
      model = transformer.create_model(params, is_train=False)
      self._load_weights_if_possible(
          model, tf.train.latest_checkpoint(self.flags_obj.model_dir))
      model.summary()

    N = 1024
    N //= params['batch_size']
    ds = self._create_dataset(params['data_dir'], batch_size=params['batch_size'], repeat=1)
    targets = []
    for (batch, ((inp, tar),)) in enumerate(ds.take(N)):
      for (index, ids) in enumerate(tar):
        real_title = self._trim_and_decode(ids)
        #print('{}: {}'.format(batch*params['batch_size'] + index, real_title))
        targets.append(real_title)
    print('load {} examples from {}'.format(len(targets), params['data_dir']))

    #numpy.set_printoptions(threshold=sys.maxsize)

    correct, total = 0, 0
    ds = ds.map(lambda X: X[0]).take(N)
    ret = model.predict(ds)
    val_outputs, _ = ret
    length = len(val_outputs)
    for i in range(length):
      pred  = self._trim_and_decode(val_outputs[i])
      target = targets[i]
      if pred == target:
        correct += 1
        #print('match #{}: \n    "{}"\n    "{}"'.format(i, pred, target))
      else:
        print('mismatch #{}: \n\tPred.:  "{}"\n\tTarget: "{}"'.format(i, pred, target))
        #print('val_outputs[i] : {0}/{1}'.format(len(val_outputs[i]), val_outputs[i]))
      total += 1
    print('accuracy: {}/{}={}'.format(correct, total, correct/total))

  def _create_callbacks(self, cur_log_dir, init_steps, params):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
    callbacks = misc.get_callbacks()
    callbacks.append(scheduler_callback)
    ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path, save_weights_only=True))
    return callbacks

  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      model.load_weights(init_weight_path).expect_partial()
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    # TODO(b/139414679): Explore the difference between using
    # LearningRateSchedule and callback for GPU runs, and try to merge them.
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule if self.use_tpu else params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    if params["dtype"] == tf.float16:
      opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          opt, loss_scale=flags_core.get_loss_scale(self.flags_obj,
                                                    default_for_fp16="dynamic"))
    if self.flags_obj.fp16_implementation == "graph_rewrite":
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    return opt

  def _create_dataset(self, dtitle_file, batch_size=None, max_body_length=1024, max_title_length=48, shuffle_size=None, repeat=1):
    def _data_encode(ln):
      _, tar, inp = tf.strings.split(ln, '\t')
      inp = self.tokenizer.encode(inp.numpy())[:max_body_length-1] + [self.EOS_id]
      tar = self.tokenizer.encode(tar.numpy()) + [self.EOS_id]
      return inp, tar

    ds = tf.data.TextLineDataset(dtitle_file)
    ds = ds.map(lambda ln: tf.py_function(_data_encode, (ln,), [tf.int64, tf.int64]))
    ds = ds.filter(lambda body, title: tf.size(title) <= max_title_length)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size)
    if batch_size:
      ds = ds.padded_batch(batch_size, padded_shapes=([max_body_length], [max_title_length]), drop_remainder=True)
    ds = ds.repeat(repeat)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x, y: ((x, y), ))

    return ds

  @classmethod
  def _ensure_dir(cls, log_dir):
    """Makes log dir if not existed."""
    if not tf.io.gfile.exists(log_dir):
      tf.io.gfile.makedirs(log_dir)


def main_test(_):
  flags_obj = flags.FLAGS
  task = TransformerTask(flags_obj)
  ds = task._create_dataset(flags_obj.data_dir, batch_size=4)
  for (batch, ((inp, tar),)) in enumerate(ds.take(2)):
    for (index, (inp1, tar1)) in enumerate(zip(inp, tar)):
      htmlbody = task._trim_and_decode(inp1)
      title = task._trim_and_decode(tar1)
      print('{}:\ninp = {}\ntar = {}\ninp_str = {}\ntar_str = {}'.format(batch*4+index, inp1, tar1, htmlbody, title))


def main(_):
  flags_obj = flags.FLAGS
  with logger.benchmark_context(flags_obj):
    task = TransformerTask(flags_obj)
    if flags_obj.mode == "train":
      task.train()
    elif flags_obj.mode == "predict":
      task.predict()
    elif flags_obj.mode == "eval":
      task.eval()
    else:
      raise ValueError("Invalid mode {}".format(flags_obj.mode))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  misc.define_transformer_flags()
  app.run(main)
  #app.run(main_test)

