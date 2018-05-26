# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""trains a seq2seq model.

based on "Abstractive Text Summarization using Sequence-to-sequence RNNS and Beyond"

"""
import sys
import os
import time
import datetime

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model

# Install pythonrouge by:
#   pip install git+https://github.com/tagucci/pythonrouge.git
from pythonrouge.pythonrouge import Pythonrouge
from pprint import pprint

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 1200, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'The mini-batch size for training.')
tf.app.flags.DEFINE_string('tf_device', None, 'default tf.device placement instruction.')
tf.app.flags.DEFINE_integer('decode_train_step', None, 'specify a train_step for the decode procedure.')


def calculate_rouge_scores(summaries, references, printScores=True, root=None, global_step=None):
  rouge = Pythonrouge(summary_file_exist=False,
                        summary=summaries, reference=references,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=True, length=5,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
  score = rouge.calc_score()

  if printScores:
    print('ROUGE-N(1-2) & ROUGE-L recall and F value:')
    pprint(score)

  if root is not None and global_step is not None:
    for key in ['ROUGE-1-R','ROUGE-2-R','ROUGE-L-R']:
      swriter = tf.summary.FileWriter(os.path.join(root, key))
      summary = tf.Summary(value=[tf.Summary.Value(tag='ROUGE(recall)', simple_value=score[key])])
      swriter.add_summary(summary, global_step)
      swriter.close()

def _Train(model, data_batcher):
  """Runs model training."""
  print(datetime.datetime.now(), '- build the model graph')
  model.build_graph()

  ckpt_saver = tf.train.Saver(keep_checkpoint_every_n_hours=12, max_to_keep=3)
  ckpt_timer = tf.train.SecondOrStepTimer(every_secs=FLAGS.checkpoint_secs)

  with tf.Session() as sess:
    # initialize or restore model
    ckpt_path = tf.train.latest_checkpoint(FLAGS.log_root)
    if ckpt_path is None:
      print(datetime.datetime.now(), '- initialize variables')
      _ = sess.run(tf.global_variables_initializer())
      summary_writer = tf.summary.FileWriter(FLAGS.log_root, graph=sess.graph)
    else:
      print(datetime.datetime.now(), '- restore model from', ckpt_path)
      ckpt_saver.restore(sess, ckpt_path)
      summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    global_step = sess.run(model.global_step)
    for _ in range(global_step): _ = data_batcher.NextBatch()

    # main loop
    last_timestamp = time.time()
    print(datetime.datetime.now(), '- start of training at global_step', global_step)
    while global_step < FLAGS.max_run_steps:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
        loss_weights, _, _) = data_batcher.NextBatch()

      (_, summary, _, global_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)

      if global_step <= 10 or global_step <= 100 and global_step % 10 == 0 or global_step % 1000 == 0:
        print(datetime.datetime.now(), '- global_step', global_step, 'is done')

      if ckpt_timer.should_trigger_for_step(global_step):
        ckpt_saver.save(sess, os.path.join(FLAGS.log_root, 'model.ckpt'), global_step=global_step)
        ckpt_timer.update_last_triggered_step(global_step)

      # write summaries
      summary_writer.add_summary(summary, global_step)
      elapsed_time = time.time() - last_timestamp
      last_timestamp += elapsed_time
      steps_per_sec = 1 / elapsed_time if elapsed_time > 0. else float('inf')
      summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='global_step/sec', simple_value=steps_per_sec)]), global_step)

    ckpt_saver.save(sess, os.path.join(FLAGS.log_root, 'model.ckpt'), global_step=global_step)
    print(datetime.datetime.now(), '- end of training at global_step', global_step)


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_root, 'eval'))
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    (article_batch, abstract_batch, targets, article_lens, abstract_lens,
     loss_weights, _, _) = data_batcher.NextBatch()
    (summaries, _, train_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    tf.logging.info(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, train_step)
    if train_step % 100 == 0:
      summary_writer.flush()


def _Infer(model, data_batcher, global_step = None):
  """Runs model training."""
  print(datetime.datetime.now(), '- build the model graph')
  model.build_graph()

  decode_root = os.path.join(FLAGS.log_root, 'decode')
  if not os.path.exists(decode_root):
    os.mkdir(decode_root)
  ckpt_saver = tf.train.Saver()

  with tf.Session() as sess:
    # restore model
    ckpt_path = tf.train.latest_checkpoint(FLAGS.log_root)
    if global_step is not None:
      ckpt_path = '%s-%d' % (ckpt_path.split('-')[0], global_step)
    print(datetime.datetime.now(), '- restore model from', ckpt_path)
    ckpt_saver.restore(sess, ckpt_path)

    global_step = int(ckpt_path.split('-')[-1])
    result_file = os.path.join(decode_root, 'summary-%d.txt' % global_step)

    # main loop
    last_timestamp = time.time()
    print(datetime.datetime.now(), '- start of inferring at global_step', global_step)
    summaries, references = [], []
    with open(result_file, 'w') as result:
      batch_count = 0
      while global_step < FLAGS.max_run_steps:
        (article_batch, abstract_batch, targets, article_lens, abstract_lens,
          loss_weights, articles, titles) = data_batcher.NextBatch(1)
        if article_batch is None: break

        (token_ids,) = model.run_infer_step(
            sess, article_batch, abstract_batch, targets, article_lens,
            abstract_lens, loss_weights)

        tokens = [model._vocab.IdToWord(wid[0][0]) for wid in token_ids if wid[0][0] != 3]
        summary = ' '.join(tokens)
        result.write('# [%d]\nArticle = %s\nTitle   = %s\nSummary = %s\n' % (batch_count, articles[0], titles[0], summary))
        summaries.append([summary])
        references.append([titles])
        batch_count += 1

    calculate_rouge_scores(summaries, references, root=decode_root, global_step=global_step)


def main(unused_argv):
  vocab = data.Vocab(FLAGS.vocab_path, 1000000)

  batch_size = FLAGS.batch_size
  if FLAGS.mode == 'decode':
    batch_size = 1

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=batch_size,
      enc_layers=4,
      enc_timesteps=120,
      dec_timesteps=30,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=0,  # If 0, no sampled softmax.
      beam_size=FLAGS.beam_size)

  batcher = batch_reader.Batcher(
      FLAGS.data_path, vocab, hps,
      FLAGS.max_article_sentences, FLAGS.max_abstract_sentences,
      bucketing=FLAGS.use_bucketing, truncate_input=FLAGS.truncate_input)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(hps, vocab)
    _Train(model, batcher)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(hps, vocab)
    _Eval(model, batcher, vocab=vocab)
  elif hps.mode == 'decode':
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    model = seq2seq_attention_model.Seq2SeqAttentionModel(hps, vocab)
    #decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    #decoder.DecodeLoop(FLAGS.decode_train_step)
    _Infer(model, batcher, FLAGS.decode_train_step)
    print('decode done.')


def main_with_device_placement(argv):
  if FLAGS.tf_device:
    print('set default tf.device to:', FLAGS.tf_device)
    with tf.device(FLAGS.tf_device):
      main(argv)
  else:
    main(argv)


if __name__ == '__main__':
  tf.app.run(main=main_with_device_placement)

