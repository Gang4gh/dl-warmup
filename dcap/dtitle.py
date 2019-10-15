"""Train and evaluate deep title generation model.
Based on: https://github.com/tensorflow/models/tree/master/official/transformer/v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import numpy as np

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from official.transformer import compute_bleu
from official.transformer.v2 import data_pipeline
import misc
from official.transformer.v2 import optimizer
import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import keras_utils
from official.utils.misc import distribution_utils


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
    params["num_parallel_calls"] = (flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

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

    train_ds = self._create_dataset(params['data_dir'], batch_size=params["batch_size"], repeat=None)
    test_ds = self._create_dataset(params['data_dir'].replace('training', 'test'), batch_size=params["batch_size"], repeat=1)

    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()
      model.compile(opt)

    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    ckpt_mgr = tf.train.CheckpointManager(checkpoint, flags_obj.model_dir, max_to_keep=3, keep_checkpoint_every_n_hours=3)
    if ckpt_mgr.latest_checkpoint:
      model.fit((np.ones((1, 1024), np.int64), np.ones((1, 48), np.int64)), verbose=0)
      checkpoint.restore(ckpt_mgr.latest_checkpoint).assert_consumed()
      current_step = opt.iterations.numpy() - 1
      logging.info("Loaded checkpoint %s, current_step %d", ckpt_mgr.latest_checkpoint, current_step)

    model.summary()

    if current_step >= flags_obj.train_steps:
      logging.info("Reach the target train_steps({}) and exit.".format(flags_obj.train_steps))
      return None

    callbacks = self._create_callbacks(flags_obj.model_dir, current_step, params, ckpt_mgr)

    logging.info("Start train iteration at global step:{}".format(current_step))
    history = model.fit(
        train_ds,
        initial_epoch=current_step // flags_obj.steps_between_evals,
        epochs=(flags_obj.train_steps-1) // flags_obj.steps_between_evals + 1,
        steps_per_epoch=min(flags_obj.steps_between_evals, flags_obj.train_steps - current_step),
        callbacks=callbacks,
        validation_data=test_ds,
        validation_steps=flags_obj.validation_steps,
        verbose=1)
    logging.info("Train history: {}".format(history.history))
    current_step = opt.iterations.numpy() - 1
    logging.info("End train iteration at global step:{}".format(current_step))

    stats = ({
        "loss": train_loss
    } if history is None else misc.build_stats(history, callbacks))

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
        self.flags_obj.bleu_ref, self.flags_obj.vocab_file)

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

  def _create_callbacks(self, cur_log_dir, init_steps, params, ckpt_mgr):
    """Creates a list of callbacks."""
    sfunc = optimizer.LearningRateFn(params["learning_rate"],
                                     params["hidden_size"],
                                     params["learning_rate_warmup_steps"])
    callbacks = misc.get_callbacks()
    callbacks.append(optimizer.LearningRateScheduler(sfunc, init_steps))
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: ckpt_mgr.save()))
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
  FLAGS = flags.FLAGS
  task = TransformerTask(FLAGS)
  ds = task._create_dataset(FLAGS.data_dir, batch_size=4)
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

