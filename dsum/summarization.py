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
import threading
import subprocess
import logging
import tensorflow as tf

from vocab import Vocab
import seq2seq_model
import data_generic as dg

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', '', 'path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path', '', 'path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('log_root', '', 'directory for model root.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000, 'maximum number of run steps.')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 1200, 'how often to checkpoint.')
tf.app.flags.DEFINE_integer('random_seed', 17, 'a seed value for randomness.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the mini-batch size for training.')
tf.app.flags.DEFINE_integer('decode_train_step', None, 'specify a train_step for the decode procedure.')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'use only top vocab_size tokens from a .vocab file.')
tf.app.flags.DEFINE_integer('eval_rouge_interval', None, 'interval to calculate ROUGE via `make decode`.')
tf.app.flags.DEFINE_bool('enable_pointer', True, 'whether enable pointer mechanism.')
tf.app.flags.DEFINE_bool('enable_logfile', False, 'whether write logging.debug() to log files.')


def setup_logging():
  class ElapsedFormatter():
    def __init__(self):
      self.start_time = time.time()
    def format(self, record):
      elapsed = int(record.created - self.start_time)
      return '+%dd %02d:%02d:%02d | %s' % (elapsed / 3600 / 24, elapsed / 3600, elapsed / 60 % 60, elapsed % 60, record.getMessage())

  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  stdout_handler.setFormatter(ElapsedFormatter())
  handlers = [stdout_handler]
  if FLAGS.enable_logfile:
    if not os.path.exists(FLAGS.log_root):
      os.mkdir(FLAGS.log_root)
    log_file = os.path.join(FLAGS.log_root, 'log-%d.txt' % int(time.time()))
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s] %(filename)s#%(lineno)d %(levelname)-5s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    handlers.append(file_handler)
  logging.basicConfig(
      level = logging.DEBUG,
      handlers = handlers
  )
  logging.getLogger('tensorflow').propagate = False
  tf.logging.set_verbosity(tf.logging.WARN)
  logging.debug('commandline: %s' % ' '.join(sys.argv))

def calculate_rouge_scores(summaries, references, max_length, root=None, global_step=None):
  # install pythonrouge by: pip install git+https://github.com/tagucci/pythonrouge.git
  from pythonrouge.pythonrouge import Pythonrouge

  logging.info('calculate ROUGE scores of %d summaries' % len(summaries))
  rouge = Pythonrouge(summary_file_exist=False,
                        summary=summaries, reference=references,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=max_length is not None, length=max_length,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
  score = rouge.calc_score()
  logging.info('ROUGE(1/2/L) F1 Scores:')
  logging.info('>   ROUGE-1-F: %f' % score['ROUGE-1-F'])
  logging.info('>   ROUGE-2-F: %f' % score['ROUGE-2-F'])
  logging.info('>   ROUGE-L-F: %f' % score['ROUGE-L-F'])

  if root is not None and global_step is not None:
    for key in ['ROUGE-1-F','ROUGE-2-F','ROUGE-L-F']:
      swriter = tf.summary.FileWriter(os.path.join(root, key))
      summary = tf.Summary(value=[tf.Summary.Value(tag='ROUGE', simple_value=score[key])])
      swriter.add_summary(summary, global_step)
      swriter.close()

def prepare_session_config():
  return tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


def _Train(model, data_filepath):
  """Runs model training."""

  logging.info('build the model graph')
  model.build_graph()
  logging.debug('tf.trainable_variables:')
  for var in tf.trainable_variables(): logging.debug('  %s' % var)

  ckpt_saver = tf.train.Saver(keep_checkpoint_every_n_hours=12, max_to_keep=3)
  ckpt_timer = tf.train.SecondOrStepTimer(every_secs=FLAGS.checkpoint_secs)

  with tf.Session(config=prepare_session_config()) as sess:
    # initialize or restore model
    ckpt_path = tf.train.latest_checkpoint(FLAGS.log_root)
    if ckpt_path is None:
      logging.info('initialize model variables')
      _ = sess.run(tf.global_variables_initializer())
      summary_writer = tf.summary.FileWriter(FLAGS.log_root, graph=sess.graph)
      model.initialize_dataset(sess, data_filepath)
    else:
      logging.info('restore model from %s', ckpt_path)
      ckpt_saver.restore(sess, ckpt_path)
      summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    global_step = sess.run(model.global_step) - 1

    # main loop
    last_timestamp, last_step = time.time(), global_step
    logging.info('start of training at global_step %d', global_step + 1)
    while global_step < FLAGS.max_run_steps:
      (_, summary, _, global_step) = model.run_train_step(sess)

      if global_step <= 10 or global_step <= 100 and global_step % 10 == 0 or global_step % 1000 == 0:
        elapsed_time = time.time() - last_timestamp
        speed = elapsed_time / max(1, global_step - last_step)
        last_timestamp, last_step = last_timestamp + elapsed_time, global_step
        logging.info('finish global_step %d, speed = %f sec/step', global_step, speed)

      if ckpt_timer.should_trigger_for_step(global_step):
        ckpt_saver.save(sess, os.path.join(FLAGS.log_root, 'model.ckpt'), global_step=global_step)
        ckpt_timer.update_last_triggered_step(global_step)

      summary_writer.add_summary(summary, global_step)

    ckpt_saver.save(sess, os.path.join(FLAGS.log_root, 'model.ckpt'), global_step=global_step)
    logging.info('end of training at global_step %d', global_step)


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_root, 'eval'))
  sess = tf.Session(config=prepare_session_config())
  while True:
    time.sleep(60)
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


def _Infer(model, data_filepath, global_step=None):
  """Runs model training."""
  logging.info('build the model graph')
  model.build_graph()

  decode_root = os.path.join(FLAGS.log_root, 'decode')
  if not os.path.exists(decode_root):
    os.mkdir(decode_root)
  ckpt_saver = tf.train.Saver()

  with tf.Session(config=prepare_session_config()) as sess:
    # restore model
    ckpt_path = tf.train.latest_checkpoint(FLAGS.log_root)
    if global_step is not None:
      ckpt_path = '%s-%d' % (ckpt_path.split('-')[0], global_step)
    logging.info('restore model from %s', ckpt_path)
    ckpt_saver.restore(sess, ckpt_path)

    model.initialize_dataset(sess, data_filepath)

    global_step = int(ckpt_path.split('-')[-1])
    result_file = os.path.join(decode_root, 'summary-%06d-%d.txt' % (global_step, int(time.time())))

    # main loop
    logging.info('begin of inferring at global_step %d', global_step)
    summaries, references = [], []
    with open(result_file, 'w', encoding='utf-8') as result:
      batch_count = 0
      while True:
        try:
          (token_ids, article_strings, summary_strings) = model.run_infer_step(sess)
          article_strings = [line.decode() for line in article_strings]
          summary_strings = [line.decode() for line in summary_strings]
        except tf.errors.OutOfRangeError:
          break

        for i in range(len(token_ids)):
          article_words = article_strings[i].split()
          tokens_rouge = [model._vocab.get_word_by_id(wid, reference=article_words)
            for wid in token_ids[i] if wid != model._vocab.token_end_id]
          tokens_print = [model._vocab.get_word_by_id(wid, reference=article_words, markup=True)
            for wid in token_ids[i] if wid != model._vocab.token_end_id]
          summary = ' '.join(tokens_print)
          result.write('# [%d]\nArticle = %s\nTitle   = %s\nSummary = %s\n'
              % (batch_count * FLAGS.batch_size + i, article_strings[i], summary_strings[i], summary))
          model_summary = ' '.join(tokens_rouge).split(dg.TOKEN_EOS_SPACES)
          refer_summary = [summary_strings[i].split(dg.TOKEN_EOS_SPACES)]
          summaries.append(model_summary)
          references.append(refer_summary)

        batch_count += 1
        if batch_count % 40 == 0:
          logging.info('finished batch_count = %d', batch_count)
    logging.info('end of inferring at global_step %d', global_step)
    calculate_rouge_scores(summaries, references, model._hps.dec_timesteps, root=decode_root, global_step=global_step)


def _naive_baseline(model, data_filepath, sentence_count=3):
  """ dump inputs from tensorflow dataset api and measure the naive baseline (first N sentences) """
  logging.info('setup model input')
  model._setup_model_input()

  with tf.Session(config=prepare_session_config()) as sess:
    model.initialize_dataset(sess, data_filepath)

    result_file = os.path.join(FLAGS.log_root, 'naive-head-%d-log.txt' % sentence_count)
    summaries, references = [], []
    with open(result_file, 'w', encoding='utf-8') as result:
      batch_count = 0
      while True:
        try:
          (article_strings, summary_strings, article_tokens, summary_tokens) = model.read_inputs(sess)
          article_strings = [line.decode() for line in article_strings]
          summary_strings = [line.decode() for line in summary_strings]
        except tf.errors.OutOfRangeError:
          break

        for i in range(len(article_strings)):
          result.write('# [%d]\nArticle = %s\nArticle Tokens = %s\nSummary = %s\nSummary Tokens = %s\n'
              % (batch_count * FLAGS.batch_size + i,
                 article_strings[i], ' '.join([str(id) for id in article_tokens[i] if id != model._vocab.token_pad_id]),
                 summary_strings[i], ' '.join([str(id) for id in summary_tokens[i] if id != model._vocab.token_pad_id]),
                 ))
          naive_summary = article_strings[i].split(dg.TOKEN_EOS_SPACES)[:sentence_count]
          refer_summary = [summary_strings[i].split(dg.TOKEN_EOS_SPACES)]
          result.write('Refer Summary = %s\nNaive Summary = %s\n' % (refer_summary, naive_summary))
          summaries.append(naive_summary)
          references.append(refer_summary)
        batch_count += 1
    calculate_rouge_scores(summaries, references, model._hps.dec_timesteps)


def check_progress_periodically(warmup_delay, check_interval):
  # when run in philly, default data/vocab/log paths in Makefile are incorrect, so set the correct values in ARGS
  ARGS = '--data_path=%s --vocab_path=%s --log_root=%s' % (
      FLAGS.data_path.replace('training.articles', 'test-sample.articles'),
      FLAGS.vocab_path, FLAGS.log_root)
  time.sleep(warmup_delay)
  while True:
    start_time = datetime.datetime.now()
    res = subprocess.run('make decode "ARGS=%s"' % ARGS, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pairs = [line.split()[-2:] for line in res.stdout.decode().split('\n') if line]
    scores = [p[1] for p in pairs if p[0].startswith('ROUGE-')]
    if len(scores) == 3:
      global_step = next((pair for pair in pairs if pair[0] == 'global_step'), [0, -1])[1]
      logging.info('ROUGE(1/2/L) = %s at global_step = %s', ' / '.join(scores), global_step)
    else:
      logging.error('Error when calculate ROUGE')
      logging.debug('Error stdout = %s', res.stdout.decode())
      logging.debug('Error stderr = %s', res.stderr.decode())
    timedelta = datetime.datetime.now() - start_time
    time.sleep(check_interval - timedelta.total_seconds() % check_interval)


def main(unused_argv):
  setup_logging()
  tf.set_random_seed(FLAGS.random_seed)

  hps = seq2seq_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      batch_size=FLAGS.batch_size,
      enc_layers=4,
      enc_timesteps=400,
      dec_timesteps=100,
      num_hidden=256,   # for rnn cell
      emb_dim=128,      # If 0, don't use embedding
      beam_size=FLAGS.beam_size)

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size, hps.enc_timesteps if FLAGS.enable_pointer else 0)
  model = seq2seq_model.Seq2SeqAttentionModel(hps, vocab)

  if hps.mode == 'train':
    # start a thread to check progress then start training
    if FLAGS.eval_rouge_interval is not None:
      threading.Thread(target=check_progress_periodically, args=(3*60, FLAGS.eval_rouge_interval), daemon=True).start()
    _Train(model, FLAGS.data_path)
  elif hps.mode == 'eval':
    _Eval(model, vocab=vocab)
  elif hps.mode == 'decode':
    _Infer(model, FLAGS.data_path, FLAGS.decode_train_step)
  elif hps.mode == 'naive':
    _naive_baseline(model, FLAGS.data_path)


if __name__ == '__main__':
  tf.app.run()
