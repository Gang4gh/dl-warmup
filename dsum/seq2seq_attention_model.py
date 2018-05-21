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

"""Sequence-to-Sequence with attention model for text summarization.
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf

HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples')


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    """function that feed previous model output rather than ground truth."""
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev
  return loop_function


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

  def run_decode_step(self, sess, article_batch, abstract_batch, targets,
                      article_lens, abstract_lens, loss_weights):
    to_return = [self._outputs, self.global_step]
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
    vsize = self._vocab.NumIds()

    uniform_initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=123)

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

      with tf.variable_scope('decoder'):
        #attention = tf.contrib.seq2seq.LuongAttention(hps.num_hidden, emb_memory, memory_sequence_length=article_lens)
        cell_decoder = tf.contrib.rnn.LSTMCell(hps.num_hidden, initializer=uniform_initializer)
        #cell_decoder = tf.contrib.seq2seq.AttentionWrapper(cell_decoder, attention, attention_layer_size=hps.num_hidden)
        helper = tf.contrib.seq2seq.TrainingHelper(emb_decoder_inputs, abstract_lens, time_major=True)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell = cell_decoder,
          helper = helper,
          initial_state = fw_state)
          #initial_state = cell_decoder.zero_state(hps.batch_size, tf.float32).clone(cell_state=fw_state))
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)

      with tf.variable_scope('output_projection'):
        projection_layer = tf.layers.Dense(vsize, use_bias=True)

      with tf.variable_scope('output'):
        model_outputs = projection_layer(outputs.rnn_output)

      if hps.mode == 'decode':
        with tf.variable_scope('decode_output'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          self._outputs = tf.concat(
              axis=1, values=[tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs])

          self._topk_log_probs, self._topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size*2)

      with tf.variable_scope('loss'):
        max_lens = tf.shape(model_outputs)[0]
        self._loss = tf.contrib.seq2seq.sequence_loss(model_outputs, targets[:max_lens, :], loss_weights[:max_lens, :])
        tf.summary.scalar('loss', tf.minimum(12.0, self._loss))

  def _add_train_op(self):
    self._train_op = tf.train.AdamOptimizer().minimize(self._loss, global_step=self.global_step, name='train_op')

  def encode_top_state(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run([self._enc_top_states, self._dec_in_state],
                       feed_dict={self._articles: enc_inputs,
                                  self._article_lens: enc_len})
    return results[0], tf.contrib.rnn.LSTMStateTuple(results[1].c[0], results[1].h[0])

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    """Return the topK results and new decoder states."""
    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    c = np.array([state.c for state in dec_init_states])
    h = np.array([state.h for state in dec_init_states])
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(c, h)

    feed = {
        self._enc_top_states: enc_top_states,
        self._dec_in_state:
            new_dec_in_state,
        self._abstracts:
            np.transpose(np.array([latest_tokens])),
        self._abstract_lens: np.ones([beam_size], np.int32)}

    ids, probs, states = sess.run(
        [self._topk_ids, self._topk_log_probs, self._dec_out_state],
        feed_dict=feed)

    new_states = [tf.contrib.rnn.LSTMStateTuple(states.c[i], states.h[i]) for i in range(beam_size)]
    return ids, probs, new_states

  def build_graph(self):
    self._add_placeholders()
    self._add_seq2seq()
    self.global_step = tf.train.get_or_create_global_step()
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
