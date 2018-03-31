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

"""Module for decoding."""

import os
import time

import beam_search
import data
from six.moves import xrange
import tensorflow as tf

from pythonrouge.pythonrouge import Pythonrouge
from pprint import pprint

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000,
                            'Number of decoding steps.')
tf.app.flags.DEFINE_integer('decode_batches_per_ckpt', 8000,
                            'Number of batches to decode before restoring next '
                            'checkpoint')

DECODE_LOOP_DELAY_SECS = 5

def evaluate_inmemory(summaries, references):
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summaries, reference=references,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=True, length=5,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print('ROUGE-N(1-2) & ROUGE-L recall and F value:')
    pprint(score)

class DecodeIO(object):
  """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    self._cnt = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.mkdir(self._outdir)
    self._result_file = None

  def Write(self, reference, decode):
    """Writes the reference and decoded outputs to RKV files.

    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """
    self._result_file.write('%s\t%s\n' % (decode, reference))
    self._summary.append([decode])
    self._reference.append([[reference]])
    self._cnt += 1
    #if self._cnt % 100 == 0:
    #  print('decoded {} articles'.format(self._cnt))

  def ResetFiles(self):
    """Resets the output files. Must be called once before Write()."""
    if self._result_file: self._result_file.close()
    timestamp = int(time.time())
    self._result_file = open(
        os.path.join(self._outdir, 'res%d'%timestamp), 'w')
    self._summary = []
    self._reference = []

  def Close(self):
    """Close the output file."""
    if self._result_file: self._result_file.close()
    evaluate_inmemory(self._summary, self._reference)


class BSDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batch_reader, hps, vocab):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      batch_reader: The batch data reader.
      hps: Hyperparamters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._hps = hps
    self._vocab = vocab
    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(os.path.join(FLAGS.log_root, 'decode'))

  def DecodeLoop(self):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    time.sleep(DECODE_LOOP_DELAY_SECS)
    self._Decode(self._saver, sess)

  def _Decode(self, saver, sess):
    """Restore a checkpoint and decode it.

    Args:
      saver: Tensorflow checkpoint saver.
      sess: Tensorflow session.
    Returns:
      If success, returns true, otherwise, false.
    """
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
      return False

    tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(
        FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)

    print('load model from checkpoint path', ckpt_path)

    self._decode_io.ResetFiles()
    while True:
      (article_batch, _, _, article_lens, _, _, origin_articles,
       origin_abstracts) = self._batch_reader.NextBatch(1)
      if article_batch is None:
        break
      for i in xrange(1):
        bs = beam_search.BeamSearch(
            self._model, self._hps.batch_size,
            self._vocab.WordToId(data.SENTENCE_START),
            self._vocab.WordToId(data.SENTENCE_END),
            self._hps.dec_timesteps)

        best_beam = bs.BeamSearch(sess, [article_batch[0]] * self._hps.batch_size, [article_lens[0]] * self._hps.batch_size)[0]
        decode_output = [int(t) for t in best_beam.tokens[1:]]
        self._DecodeBatch(
            origin_articles[i], origin_abstracts[i], decode_output)
    self._decode_io.Close()
    return True

  def _DecodeBatch(self, article, abstract, output_ids):
    """Convert id to words and writing results.

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      output_ids: The abstract word ids output by machine.
    """
    decoded_output = ' '.join(data.Ids2Words(output_ids, self._vocab))
    end_p = decoded_output.find(data.SENTENCE_END, 0)
    if end_p != -1:
      decoded_output = decoded_output[:end_p]
    tf.logging.info('article:  %s', article)
    tf.logging.info('abstract: %s', abstract)
    tf.logging.info('decoded:  %s', decoded_output)
    self._decode_io.Write(abstract, decoded_output.strip())
