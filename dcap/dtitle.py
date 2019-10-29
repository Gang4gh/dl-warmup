"""Train and evaluate deep title generation model.
Based on: https://github.com/tensorflow/models/tree/master/official/transformer/v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

import misc
from official.transformer.v2 import optimizer
import transformer
from official.transformer.v2 import translate
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.utils.misc import distribution_utils
import metrics

class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

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
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training

    self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(self.flags_obj.vocab_file)
    self.EOS_id = self.tokenizer.encode('<EOS>')[0]
    params["vocab_size"] = self.tokenizer.vocab_size
    logging.info('loaded vocab from {}, vocab_size={} and EOS_id={}'.format(self.flags_obj.vocab_file, self.tokenizer.vocab_size, self.EOS_id))

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

  def train(self):
    """Trains the model."""
    params = self.params
    flags_obj = self.flags_obj
    # Sets config options.
    keras_utils.set_session_config(
        enable_xla=flags_obj.enable_xla)

    train_ds = self._create_dataset(params['data_dir'], batch_size=params["batch_size"], repeat=None)
    test_ds = self._create_dataset(params['data_dir'].replace('training', 'test'),
        batch_size=params["batch_size"], repeat=1)

    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      model = transformer.create_model(params, is_train=True)
      opt = self._create_optimizer()
      loss_fn = self._create_loss_fn()
      model.compile(optimizer=opt, loss=loss_fn)
      model.summary()

    self._ensure_dir(flags_obj.model_dir)
    current_step = 0
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    ckpt_mgr = tf.train.CheckpointManager(checkpoint, flags_obj.model_dir, max_to_keep=5, keep_checkpoint_every_n_hours=12)
    if ckpt_mgr.latest_checkpoint:
      model.fit((np.ones((1, params['max_input_length']), np.int64), np.ones((1, params['max_target_length']), np.int64)), verbose=0)
      checkpoint.restore(ckpt_mgr.latest_checkpoint).assert_consumed()
      current_step = opt.iterations.numpy() - 1
      logging.info("Loaded checkpoint %s, current_step %d", ckpt_mgr.latest_checkpoint, current_step)

    if current_step >= flags_obj.train_steps:
      logging.info("Reach the target train_steps({}) and exit.".format(flags_obj.train_steps))
      return None

    logging.info("Start train iteration at global step:{}".format(current_step))
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
    current_step = opt.iterations.numpy() - 1
    logging.info("End train iteration at global step:{}".format(current_step))

    return history

  def eval(self):
    """Evaluates the model."""
    with distribution_utils.get_strategy_scope(self.distribution_strategy):
      if not self.predict_model:
        self.predict_model = transformer.create_model(self.params, is_train=True)
        self.predict_model.compile()
        self.predict_model.summary()
      self._load_model_weights(self.predict_model)

    N = 128
    ds = self._create_dataset(self.params['data_dir'], batch_size=self.params["batch_size"], repeat=1)
    res = self.predict_model.evaluate(ds, steps=N)
    print('evaluate {} steps: {}'.format(N, res))

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

    model = transformer.create_model(params, is_train=False)
    model.summary()
    self._load_model_weights(model)

    N = 1024
    N //= params['batch_size']
    ds = self._create_dataset(params['data_dir'], batch_size=params['batch_size'], repeat=1)
    targets = []
    for ((inp, tar),) in ds.take(N):
      for ids in tar:
        real_title = self._trim_and_decode(ids)
        targets.append(real_title)
    logging.info('load {} examples from {}'.format(len(targets), params['data_dir']))

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
        logging.info('mismatch #{}:  Pred.:  "{}" | Target: "{}"'.format(i, pred, target))
        #print('val_outputs[i] : {0}/{1}'.format(len(val_outputs[i]), val_outputs[i]))
      total += 1
    logging.info('the accuracy: {}/{}={}'.format(correct, total, correct/total))

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
      logging.info("load model weights from: {}".format(checkpoint_path))
      checkpoint.restore(checkpoint_path).expect_partial()
    else:
      logging.info('no checkpoint found from: {}'.format(checkpoint_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    opt = tf.keras.optimizers.Adam(
        params["learning_rate"],
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    return opt

  def _create_loss_fn(self):
    params = self.params
    if self.flags_obj.loss_fn == 'smoothed_cross_entropy':
      label_smoothing = params["label_smoothing"]
      vocab_size = params["vocab_size"]
      print('use smoothed_cross_entropy')
      def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [params['batch_size'], -1])
        y_pred = tf.reshape(y_pred, [params['batch_size'], -1, vocab_size])
        return metrics.transformer_loss(y_pred, y_true, label_smoothing, vocab_size)
      return loss
    else:
      return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  def _create_dataset(self, dtitle_file, batch_size, repeat, shuffle_size=None):
    max_input_length = self.params['max_input_length']
    max_target_length = self.params['max_target_length']

    def _data_encode(ln):
      _, tar, inp = tf.strings.split(ln, '\t')
      inp = self.tokenizer.encode(inp.numpy())[:max_input_length-1] + [self.EOS_id]
      tar = self.tokenizer.encode(tar.numpy()) + [self.EOS_id]
      return inp, tar

    ds = tf.data.TextLineDataset(dtitle_file)
    ds = ds.map(lambda ln: tf.py_function(_data_encode, (ln,), [tf.int64, tf.int64]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.filter(lambda body, title: tf.size(title) <= max_target_length)
    if shuffle_size:
      ds = ds.shuffle(shuffle_size)
    ds = ds.padded_batch(batch_size, padded_shapes=([max_input_length], [max_target_length]), drop_remainder=True)
    ds = ds.map(lambda x, y: ((x, y), y))
    ds = ds.repeat(repeat)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

  @classmethod
  def _ensure_dir(cls, log_dir):
    """Makes log dir if not existed."""
    if not tf.io.gfile.exists(log_dir):
      tf.io.gfile.makedirs(log_dir)


def main_test(_):
  FLAGS = flags.FLAGS
  task = TransformerTask(FLAGS)
  ds = task._create_dataset(FLAGS.data_dir, batch_size=4, repeat=1)
  for (batch, ((inp, tar),)) in enumerate(ds.take(2)):
    for (index, (inp1, tar1)) in enumerate(zip(inp, tar)):
      htmlbody = task._trim_and_decode(inp1)
      title = task._trim_and_decode(tar1)
      print('{}:\ninp = {}\ntar = {}\ninp_str = {}\ntar_str = {}'.format(batch*4+index, inp1, tar1, htmlbody, title))


def main(_):
  flags_obj = flags.FLAGS
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

