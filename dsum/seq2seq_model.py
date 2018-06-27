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

"""sequence-to-Sequence with attention model"""

import logging
import numpy as np
import tensorflow as tf

from collections import namedtuple
HParams = namedtuple('HParams',
                     'mode batch_size '
                     'enc_layers enc_timesteps dec_timesteps '
                     'num_hidden emb_dim '
                     'beam_size')


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def run_train_step(self, sess):
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(to_return)

  def run_eval_step(self, sess, article_batch, abstract_batch, targets,
                    article_lens, abstract_lens, loss_weights):
    to_return = [self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def run_infer_step(self, sess):
    to_return = [self._predicted_ids, self._article_strings, self._summary_strings]
    return sess.run(to_return)

  def read_inputs(self, sess):
    """read inputs from dataset for naive baseline or test purpose.
    returns:
      article_strings: string of [batch_size]
      summary_strings: string of [batch_size]
      articles: int32 of [batch_size, max_encoding_len]
      targets: int32 of [batch_size, max_decoding_len]
    """
    to_return = [self._article_strings, self._summary_strings, self._articles, self._targets]
    return sess.run(to_return)

  def initialize_dataset(self, sess, data_filepath):
    sess.run(self._iterator.initializer,
             feed_dict = {self._data_filepath: data_filepath})

  def _setup_model_input(self):
    hps = self._hps
    pad_id = self._vocab.token_pad_id
    start_id = self._vocab.token_start_id
    end_id = self._vocab.token_end_id

    def _parse_line(line):
      article_ids, _, article_text, summary_ids, _, summary_text = self._vocab.parse_article(line.decode())
      article_len = len(article_ids)
      if article_len < hps.enc_timesteps:
        article_ids = article_ids + [pad_id] * (hps.enc_timesteps - article_len)
      else:
        article_ids = article_ids[:hps.enc_timesteps]
        article_len = hps.enc_timesteps

      summary_len = len(summary_ids)
      if summary_len <= hps.dec_timesteps - 1:
        summary_ids = [start_id] + summary_ids + [end_id] + [pad_id] * (hps.dec_timesteps - 1 - summary_len)
        summary_len += 1
      else:
        summary_ids = [start_id] + summary_ids[:hps.dec_timesteps]
        summary_len = hps.dec_timesteps

      return (
          np.array(article_ids, np.int32),
          np.int32(article_len),
          np.array(summary_ids, np.int32),
          np.int32(summary_len),
          article_text,
          summary_text,)

    def fix_shapes(article_ids, article_len, summary_ids, summary_len, article_text, summary_text):
      article_ids.set_shape([hps.enc_timesteps])
      summary_ids.set_shape([hps.dec_timesteps + 1])
      article_len.set_shape([])
      summary_len.set_shape([])
      lossmask = tf.sequence_mask(summary_len, hps.dec_timesteps, dtype=tf.float32)
      article_text.set_shape([])
      summary_text.set_shape([])
      return article_ids, summary_ids[:-1], summary_ids[1:], lossmask, article_len, summary_len, article_text, summary_text

    self._data_filepath = tf.placeholder(tf.string, shape=[])
    dataset = tf.data.TextLineDataset(self._data_filepath).prefetch(hps.batch_size * 100)
    dataset = dataset.map(lambda line: tf.py_func(_parse_line, [line], [tf.int32, tf.int32, tf.int32, tf.int32, tf.string, tf.string], stateful=False))
    dataset = dataset.map(fix_shapes)
    if hps.mode != 'decode' and hps.mode != 'naive':
      dataset = dataset.repeat()
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(hps.batch_size))
    dataset = dataset.prefetch(1)
    logging.debug('dataset shape: %s', dataset)
    self._iterator = dataset.make_initializable_iterator()

    iterator_state = tf.contrib.data.make_saveable_from_iterator(self._iterator)
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, iterator_state)

    next_res = self._iterator.get_next()
    self._articles, self._abstracts, self._targets, self._loss_weights, self._article_lens, self._abstract_lens, self._article_strings, self._summary_strings = next_res
    #self._articles = tf.reshape(self._articles, [hps.batch_size, hps.enc_timesteps])
    #self._abstracts = tf.reshape(self._abstracts, [hps.batch_size, hps.dec_timesteps])
    #self._targets = tf.reshape(self._targets, [hps.batch_size, hps.dec_timesteps])
    #self._loss_weights = tf.reshape(self._loss_weights, [hps.batch_size, hps.dec_timesteps]) #tf.float32
    #self._article_lens = tf.reshape(self._article_lens, [hps.batch_size])
    #self._abstract_lens = tf.reshape(self._abstract_lens, [hps.batch_size])

  def _add_seq2seq(self):
    hps = self._hps
    vsize = self._vocab.get_vocab_size()

    uniform_initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.variable_scope('seq2seq'):
      encoder_inputs = tf.transpose(self._articles)
      decoder_inputs = tf.transpose(self._abstracts)
      targets = tf.transpose(self._targets)
      loss_weights = tf.transpose(self._loss_weights)
      article_lens = self._article_lens
      abstract_lens = self._abstract_lens

      # Embedding shared by the input and outputs.
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim],
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
        emb_encoder_inputs = tf.nn.embedding_lookup(embedding, encoder_inputs)
        emb_decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs)
 
      for layer_i in range(hps.enc_layers):
        with tf.variable_scope('encoder%d' % layer_i):
          cell_fw = tf.contrib.rnn.LSTMCell(hps.num_hidden, initializer=uniform_initializer)
          cell_bw = tf.contrib.rnn.LSTMCell(hps.num_hidden, initializer=uniform_initializer)
          (rnn_outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(
              cell_fw, cell_bw, emb_encoder_inputs,
              sequence_length=article_lens, dtype=tf.float32, time_major=True)
          emb_encoder_inputs = tf.concat(rnn_outputs, 2)
      emb_memory = tf.transpose(emb_encoder_inputs, [1, 0, 2])

      initial_dec_state = fw_state
      #initial_dec_state = tf.layers.dense(tf.concat([fw_state, bw_state], -1), hps.num_hidden)
      #initial_dec_state = tf.contrib.rnn.LSTMStateTuple(initial_dec_state[0], initial_dec_state[1])

      projection_layer = tf.layers.Dense(vsize, use_bias=True)

      with tf.variable_scope('decoder'):
        if hps.mode != 'decode':
          cell_decoder = tf.contrib.rnn.LSTMCell(hps.num_hidden, initializer=uniform_initializer)
          attention = tf.contrib.seq2seq.LuongAttention(hps.num_hidden, emb_memory, memory_sequence_length=article_lens)
          cell_decoder = tf.contrib.seq2seq.AttentionWrapper(cell_decoder, attention, attention_layer_size=hps.num_hidden)
          helper = tf.contrib.seq2seq.TrainingHelper(emb_decoder_inputs, abstract_lens, time_major=True)
          decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = cell_decoder,
            helper = helper,
            initial_state = cell_decoder.zero_state(hps.batch_size, tf.float32).clone(cell_state=initial_dec_state))
          outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
          model_outputs = projection_layer(outputs.rnn_output, scope='decoder/dense')
          max_len = tf.shape(model_outputs)[0]
          self._loss = tf.contrib.seq2seq.sequence_loss(model_outputs, targets[:max_len, :], loss_weights[:max_len, :])
          tf.summary.scalar('loss', tf.minimum(12.0, self._loss))
        else:
          emb_memory = tf.contrib.seq2seq.tile_batch(emb_memory, multiplier=hps.beam_size)
          article_lens = tf.contrib.seq2seq.tile_batch(article_lens, multiplier=hps.beam_size)
          fw_state = tf.contrib.seq2seq.tile_batch(fw_state, multiplier=hps.beam_size)

          cell_decoder = tf.contrib.rnn.LSTMCell(hps.num_hidden, initializer=uniform_initializer)
          attention = tf.contrib.seq2seq.LuongAttention(hps.num_hidden, emb_memory, memory_sequence_length=article_lens)
          cell_decoder = tf.contrib.seq2seq.AttentionWrapper(cell_decoder, attention, attention_layer_size=hps.num_hidden)

          initial_state = cell_decoder.zero_state(hps.batch_size * hps.beam_size, tf.float32).clone(cell_state=fw_state)
          start_token_ids = tf.fill([hps.batch_size], self._vocab.token_start_id)
          end_token_id = self._vocab.token_end_id

          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell_decoder,
            embedding=embedding,
            start_tokens=start_token_ids, end_token=end_token_id,
            initial_state=initial_state,
            beam_width=hps.beam_size,
            output_layer=projection_layer)
          outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=hps.dec_timesteps, output_time_major=True)
          self._predicted_ids = tf.transpose(outputs.predicted_ids[:, :, 0])

  def _add_train_op(self):
    self._train_op = tf.train.AdamOptimizer().minimize(self._loss, global_step=self.global_step, name='train_op')

  def build_graph(self):
    self._setup_model_input()
    self._add_seq2seq()
    self.global_step = tf.train.get_or_create_global_step()
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()

