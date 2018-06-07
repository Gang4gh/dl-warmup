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

import numpy as np
import tensorflow as tf

from collections import namedtuple
HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples, beam_size')


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def run_train_step(self, sess, article_batch, abstract_batch, targets,
                     article_lens, abstract_lens, loss_weights):
    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

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

  def run_infer_step(self, sess, article_batch, abstract_batch, targets,
                      article_lens, abstract_lens, loss_weights):
    to_return = [self._predicted_ids]
    return sess.run(to_return,
                    feed_dict={self._articles: article_batch,
                               self._abstracts: abstract_batch,
                               self._targets: targets,
                               self._article_lens: article_lens,
                               self._abstract_lens: abstract_lens,
                               self._loss_weights: loss_weights})

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    self._articles = tf.placeholder(tf.int32,
                                    [hps.batch_size, hps.enc_timesteps],
                                    name='articles')
    self._abstracts = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps],
                                     name='abstracts')
    self._targets = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='targets')
    self._article_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='article_lens')
    self._abstract_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='abstract_lens')
    self._loss_weights = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='loss_weights')

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
          (rnn_outputs, (fw_state, _)) = tf.nn.bidirectional_dynamic_rnn(
              cell_fw, cell_bw, emb_encoder_inputs,
              sequence_length=article_lens, dtype=tf.float32, time_major=True)
          emb_encoder_inputs = tf.concat(rnn_outputs, 2)
      emb_memory = tf.transpose(emb_encoder_inputs, [1, 0, 2])

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
            initial_state = cell_decoder.zero_state(hps.batch_size, tf.float32).clone(cell_state=fw_state))
            #initial_state = fw_state)
          outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
          model_outputs = projection_layer(outputs.rnn_output, scope='decoder/dense')
          with tf.variable_scope('loss'):
            max_len = tf.shape(model_outputs)[0]
            model_outputs = tf.pad(model_outputs, [[0, hps.dec_timesteps - max_len], [0,0], [0,0]])
            self._loss = tf.contrib.seq2seq.sequence_loss(model_outputs, targets, loss_weights)
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
          outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=30, output_time_major=True)
          self._predicted_ids = tf.transpose(outputs.predicted_ids[:, :, 0])

  def _add_train_op(self):
    self._train_op = tf.train.AdamOptimizer().minimize(self._loss, global_step=self.global_step, name='train_op')


  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.train.get_or_create_global_step()
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()

