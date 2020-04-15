# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Misc for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-import-order
from absl import flags

from official.nlp.transformer import model_params
from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS

TRIAL_PARAMS = model_params.BASE_PARAMS.copy()
TRIAL_PARAMS.update(
    hidden_size=256,
    num_heads=4,
    filter_size=1024,
)
TRIAL_MULTI_GPU_PARAMS = TRIAL_PARAMS.copy()
TRIAL_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'tiny-multipgu': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'base-multigpu': model_params.BASE_MULTI_GPU_PARAMS,
    'big': model_params.BIG_PARAMS,
    'big-multigpu': model_params.BIG_MULTI_GPU_PARAMS,
    'trial': TRIAL_PARAMS,
    'trial-multigpu': TRIAL_MULTI_GPU_PARAMS,
}


def get_model_params(param_set, num_gpus):
  """Gets predefined model params."""
  if num_gpus > 1:
    param_set += '-multigpu'
  return PARAMS_MAP[param_set].copy()


def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, etc.).
  flags_core.define_base(num_gpu=True, distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=True,
      loss_scale=True,
      all_reduce_alg=True,
      enable_xla=True,
      force_v2_in_keras_compile=True,
      fp16_implementation=True
  )

  # Additional performance flags
  # TODO(b/76028325): Remove when generic layout optimizer is ready.
  flags.DEFINE_boolean(
      name='enable_grappler_layout_optimizer',
      default=True,
      help='Enable Grappler layout optimizer. Currently Grappler can '
           'de-optimize fp16 graphs by forcing NCHW layout for all '
           'convolutions and batch normalizations, and this flag allows to '
           'disable it.'
  )

  flags_core.define_benchmark()
  flags_core.define_device(tpu=True)

  flags.DEFINE_integer(
      name='train_steps', short_name='ts', default=300000,
      help=flags_core.help_wrap('The number of steps used to train.'))
  flags.DEFINE_integer(
      name='steps_between_evals', short_name='sbe', default=1000,
      help=flags_core.help_wrap(
          'The Number of training steps to run between evaluations. This is '
          'used if --train_steps is defined.'))
  flags.DEFINE_boolean(
      name='enable_time_history', default=True,
      help='Whether to enable TimeHistory callback.')
  flags.DEFINE_boolean(
      name='enable_tensorboard', default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_integer(
      name='steps_between_tensorboard_log', default=10,
      help=flags_core.help_wrap('The number of steps to write tensorboard log.'))
  flags.DEFINE_boolean(
      name='enable_metrics_in_training', default=False,
      help='Whether to enable metrics during training.')
  flags.DEFINE_string(
      name='profile_steps', default=None,
      help='Save profiling data to model dir at given range of steps. The '
      'value must be a comma separated pair of positive integers, specifying '
      'the first and last step to profile. For example, "--profile_steps=2,4" '
      'triggers the profiler to process 3 steps, starting from the 2nd step. '
      'Note that profiler has a non-trivial performance overhead, and the '
      'output file can be gigantic if profiling many steps.')
  # Set flags from the flags_core module as 'key flags' so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name='param_set', short_name='mp', default='big',
      enum_values=PARAMS_MAP.keys(),
      help=flags_core.help_wrap(
          'Parameter set to use when creating and training the model. The '
          'parameters define the input shape (batch size and max length), '
          'model configuration (size of embedding, # of hidden layers, etc.), '
          'and various other settings. The big parameter set increases the '
          'default batch size, embedding/hidden size, and filter size. For a '
          'complete list of parameters, please see model/model_params.py.'))

  flags.DEFINE_bool(
      name='static_batch', short_name='sb', default=False,
      help=flags_core.help_wrap(
          'Whether the batches in the dataset should have static shapes. In '
          'general, this setting should be False. Dynamic shapes allow the '
          'inputs to be grouped so that the number of padding tokens is '
          'minimized, and helps model training. In cases where the input shape '
          'must be static (e.g. running on TPU), this setting will be ignored '
          'and static batching will always be used.'))
  flags.DEFINE_integer(
      name='max_input_length', short_name='mil', default=1024,
      help=flags_core.help_wrap('Max input sequence length (token count) for Transformer'))
  flags.DEFINE_integer(
      name='max_target_length', short_name='mtl', default=48,
      help=flags_core.help_wrap('Max target sequence length (token count) for Transformer'))

  # Flags for training with steps (may be used for debugging)
  flags.DEFINE_integer(
      name='validation_example_count', short_name='vec', default=1024,
      help=flags_core.help_wrap('The number of examples used in validation.'))

  # BLEU score computation
  flags.DEFINE_string(
      name='bleu_source', short_name='bls', default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
          ))
  flags.DEFINE_string(
      name='bleu_ref', short_name='blr', default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
          ))
  flags.DEFINE_string(
      name='vocab_file', short_name='vf', default=None,
      help=flags_core.help_wrap(
          'Path to subtoken vocabulary file. If data_download.py was used to '
          'download and encode the training data, look in the data_dir to find '
          'the vocab file.'))
  flags.DEFINE_string(
      name='mode', default='train',
      help=flags_core.help_wrap('mode: train, eval, or predict'))
  flags.DEFINE_bool(
      name='use_ctl',
      default=False,
      help=flags_core.help_wrap(
          'Whether the model runs with custom training loop.'))
  flags.DEFINE_bool(
      name='use_tpu_2vm_config',
      default=False,
      help=flags_core.help_wrap(
          'Whether the model runs in 2VM mode, Headless server and unit test '
          'all use 1VM config.'))
  flags.DEFINE_integer(
      name='decode_batch_size',
      default=32,
      help=flags_core.help_wrap(
          'Global batch size used for Transformer autoregressive decoding on '
          'TPU.'))
  flags.DEFINE_integer(
      name='decode_max_length',
      default=97,
      help=flags_core.help_wrap(
          'Max sequence length of the decode/eval data. This is used by '
          'Transformer autoregressive decoding on TPU to have minimum '
          'paddings.'))
  flags.DEFINE_bool(
      name='padded_decode',
      default=False,
      help=flags_core.help_wrap(
          'Whether the autoregressive decoding runs with input data padded to '
          'the decode_max_length. For TPU/XLA-GPU runs, this flag has to be '
          'set due the static shape requirement. Although CPU/GPU could also '
          'use padded_decode, it has not been tested. In addition, this method '
          'will introduce unnecessary overheads which grow quadratically with '
          'the max sequence length.'))

  flags.DEFINE_string(
      name='loss_fn', default='smoothed_corss_entropy',
      help=flags_core.help_wrap('loss_fn: corss_entropy, smoothed_corss_entropy'))

  flags.DEFINE_string(
      name='input_concat_schema', default='v2',
      help=flags_core.help_wrap(
          'input_concat_schema: [v0, v1, v2, v3]. v0: html only; '
          'v1: concatenated (url, hostname, html); '
          'v2: concatenated and padded (url, hostname, html); '
          'v3: padded (url, hostname, html)'))

  flags.DEFINE_bool(
      name='compact_predict_result', default=False,
      help=flags_core.help_wrap('Whether dump predict result as a TSV'))

  flags.DEFINE_integer(
      name='max_predict_count',
      default=None,
      help=flags_core.help_wrap('max example count to predict'))

  flags.DEFINE_string(
      name='prediction_details_file', default=None,
      help=flags_core.help_wrap(
          'output prediction details to the specified file. '
          'disabled when None; output to the model folder when #model_dir.'))

  flags.DEFINE_string(
      name='prediction_reference_file', default=None,
      help=flags_core.help_wrap('reference file for prediction details'))

  flags.DEFINE_string(
      name='prediction_compact_file', default='#model_dir',
      help=flags_core.help_wrap(
          'output prediction compact result to the specified file, '
          'disabled when None; output to the model folder when #model_dir.'))

  flags.DEFINE_bool(
      name='calc_rouge_scores', default=True,
      help=flags_core.help_wrap('Whether to calculate ROUGE scores or not'))

  flags.DEFINE_bool(
      name='use_reformer', default=False,
      help=flags_core.help_wrap('use Reformer model instead of Transformer'))

  flags.DEFINE_bool(
      name='use_full_attention_in_reformer', default=False,
      help=flags_core.help_wrap('use full attention in reformer, instead of LSH attention, for eval purpose'))

  flags.DEFINE_integer(
      name='num_hashes',
      default=4,
      help=flags_core.help_wrap('number of hashes used in LSH attention for training'))

  flags.DEFINE_integer(
      name='test_num_hashes',
      default=None,
      help=flags_core.help_wrap('number of hashes used in LSH attention for test'))

  flags.DEFINE_integer(
      name='bucket_size',
      default=64,
      help=flags_core.help_wrap('bucket size for LSH attention'))

  flags.DEFINE_string(
      name='val_data_dir', default=None,
      help=flags_core.help_wrap('validation data file used in training. If None, then try to find matching test file based on data_dir'))

  flags.DEFINE_float(
      name='one_dropout', default=None,
      help=flags_core.help_wrap('one dropout rate for all layers'))

  flags.DEFINE_float(
      name='attention_dropout', default=None,
      help=flags_core.help_wrap('dropout rate for attention layers'))

  flags.DEFINE_float(
      name='lsh_attention_dropout', default=0.0,
      help=flags_core.help_wrap('dropout rate for lsh_attention layers'))

  flags.DEFINE_bool(
      name='dev_mode', default=False,
      help=flags_core.help_wrap('if dev_mode is True, output more details'))

  flags.DEFINE_string(
      name='training_schema', default=None,
      help=flags_core.help_wrap('format: input1:limit1,input2:limit2...=>target'))

  flags_core.set_defaults(data_dir='/tmp/translate_ende',
                          model_dir='/tmp/transformer_model',
                          batch_size=16)
