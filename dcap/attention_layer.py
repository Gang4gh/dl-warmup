# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.nlp import bert_modeling as common_layer


class Attention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads
    self.query_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="query")
    self.key_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="key")
    self.value_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="value")
    self.output_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, output_projection=True, name="output_transform")
    super(Attention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def call(self, query_input, source_input, bias, training, cache=None,
           decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]}
        where i is the current decoded length for non-padded decode, or max
        sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    query = self.query_dense_layer(query_input)
    key = self.key_dense_layer(source_input)
    value = self.value_dense_layer(source_input)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    # Scale query to prevent the dot product between query and key from growing
    # too large.
    depth = (self.hidden_size // self.num_heads)
    query *= depth ** -0.5

    # Calculate dot product attention
    logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
    logits += bias
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, name="attention_weights")
    if training:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, query_input, bias, training, cache=None,
           decode_loop_step=None):
    return super(SelfAttention, self).call(
        query_input, query_input, bias, training, cache, decode_loop_step)


class LshSelfAttention(tf.keras.layers.Layer):
  """Multi-headed LSH attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout, num_hashes, bucket_size):
    """Initialize LshSelfAttention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same LSH attention structure.
      attention_dropout: float, dropout rate inside LSH attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(LshSelfAttention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = 0#attention_dropout
    self.num_hashes = num_hashes
    self.bucket_size = bucket_size

  def build(self, input_shape):
    """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
    size_per_head = self.hidden_size // self.num_heads
    self.sharedQK_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="sharedQK")
    self.value_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, name="value")
    self.output_dense_layer = common_layer.Dense3D(
        self.num_heads, size_per_head, kernel_initializer="glorot_uniform",
        use_bias=False, output_projection=True, name="output_transform")
    super(LshSelfAttention, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
        "num_hashes": self.num_hashes,
        "bucket_size": self.bucket_size,
    }


  def call(self, query_input, padding_mask, training, cache=None,
           decode_loop_step=None):
    """Apply LSH self attention mechanism from query_input to query_input(itself).

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      padding_mask: A tensor with shape [batch_size, length_source], type=tf.bool
        the 'Ture' value means invalid position to be marked out.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]}
        where i is the current decoded length for non-padded decode, or max
        sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    query = self.sharedQK_dense_layer(query_input)
    key = tf.math.l2_normalize(query, -1)
    value = self.value_dense_layer(query_input)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape.as_list()
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = tf.concat([tf.cast(cache["k"], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache["v"], value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    ## Scale query to prevent the dot product between query and key from growing
    ## too large.
    #depth = (self.hidden_size // self.num_heads)
    #query *= depth ** -0.5
    #attention_output = calculate_full_attention(key, query, value, bias, training, self.attention_dropout)

    #attention_output = calculate_full_attention_v2(query, key, value, padding_mask, apply_soft_selfmask=True, dropout = self.attention_dropout if training else 0)

    attention_output = calculate_LSH_attention(query, value, padding_mask, num_hashes=self.num_hashes, bucket_size=self.bucket_size, dropout = self.attention_dropout if training else 0.0)

    # Run the outputs through another linear projection layer. Recombining heads
    # is automatically done --> [batch_size, length, hidden_size]
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


def calculate_full_attention(key, query, value, bias, training, attention_dropout):
  # Calculate dot product attention
  logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
  logits += bias

  # Note that softmax internally performs math operations using float32
  # for numeric stability. When training with float16, we keep the input
  # and output in float16 for better performance.
  weights = tf.nn.softmax(logits, name="attention_weights")

  if training:
    weights = tf.nn.dropout(weights, rate=attention_dropout)
  attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)
  return attention_output


def calculate_full_attention_v2(query, key, value, padding_mask=None, apply_causal_mask=False, apply_soft_selfmask=False, dropout=0.0):
  """
  Args:
    query: with shape [batch_size, query_length, num_heads, dim_per_head]
    key  : with shape [batch_size, value_length, num_heads, dim_per_head]
    value: with shape [batch_size, value_length, num_heads, dim_per_head]
    padding_mask: a mask tensor with shape [batch_size, value_length]
    apply_causal_mask: mask tokens after current token, used in self attention
    apply_soft_selfmask: mask query token itself softly, used in self attention
    dropout: attention dropout ratio

  Returns:
    Attention output with shape [batch_size, value_length, num_heads, dim_per_head]
  """
  _, value_length, _, dim_per_head = value.shape

  query *= dim_per_head ** -0.5
  logits = tf.einsum("BFNH,BTNH->BNFT", query, key)

  if padding_mask is not None:
    logits = tf.where(padding_mask[:,None,None,:], -1e9, logits)
  if apply_causal_mask:
    causal_validmask =  tf.linalg.band_part(tf.ones([value_length, value_length], dtype=tf.bool), -1, 0)[None,None,:,:]
    logits = tf.where(causal_validmask, logits, -1e9)
  if apply_soft_selfmask:
    self_mask =  tf.linalg.band_part(tf.ones([value_length, value_length], dtype=tf.bool), 0, 0)[None,None,:,:]
    logits = tf.where(self_mask, -1e5, logits)

  weights = tf.nn.softmax(logits, name="attention_weights")

  if dropout > 0:
    weights = tf.nn.dropout(weights, rate=dropout)

  attentions = tf.einsum("BNFT,BTNH->BFNH", weights, value)
  return attentions


def get_hash():
  rotated_vecs = tf.einsum('blhd,bdHi->bhlHi', vecs, random_rotations)
  rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
  buckets = tf.math.argmax(rotated_vecs, axis=-1, output_type=tf.int32)
  #buckets = tf.reshape(buckets, (batch_size, length, num_heads, -1,))

  #output shape: [batch_size, num_heads, length, num_hashes]
  return buckets


import TFefficient_attention
from TFutils import sort_key_val, batched_index_select, make_unit_length, chunked_sum, process_inputs_chunk

lsh_att = None
def calculate_LSH_attention(qk, value, padding_mask=None, dropout=0, num_hashes=2, bucket_size=64):
  global lsh_att
  if lsh_att is None:
    lsh_att = TFefficient_attention.TFLSHAttention(dropout = dropout, n_hashes=num_hashes, bucket_size=bucket_size, causal=False)

  batch_size, length, num_heads, num_dim = qk.shape
  qk = tf.reshape(tf.transpose(qk, perm=[0,2,1,3]), (-1, length, num_dim))
  value = tf.reshape(tf.transpose(value, perm=[0,2,1,3]), (-1, length, num_dim))
  #TODO: not efficient to expand padding_mask here
  padding_mask = tf.keras.backend.repeat_elements(padding_mask, rep=num_heads, axis=0)
  ret = lsh_att.call(qk, value, padding_mask)
  ret = tf.transpose(tf.reshape(ret, (batch_size, num_heads, length, num_dim)), perm=[0,2,1,3])
  return ret

def calculate_LSH_attention_v2(qk, value, padding_mask=None, dropout=0, num_hashes=2, bucket_size=64):
  #TODO: rewrite and clean up LSH attention
  global lsh_att
  if lsh_att is None:
    lsh_att = TFefficient_attention.TFLSHAttention(dropout = dropout, n_hashes=num_hashes, bucket_size=bucket_size, causal=False)

  batch_size, length, num_heads, num_dim = qk.shape
  qk = tf.reshape(tf.transpose(qk, perm=[0,2,1,3]), (-1, length, num_dim))
  value = tf.reshape(tf.transpose(value, perm=[0,2,1,3]), (-1, length, num_dim))
  #TODO: not efficient to expand padding_mask here
  ret = lsh_att(qk, value, tf.keras.backend.repeat_elements(padding_mask, rep=num_heads, axis=0))
  ret = tf.transpose(tf.reshape(ret, (batch_size, num_heads, length, num_dim)), perm=[0,2,1,3])
  return ret

def get_self_attention_mask(length, dtype=tf.float32):
	with tf.name_scope("self_attention_bias"):
		self_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype), 0, 0)
		self_locs = tf.reshape(self_locs, [1, 1, length, length])
	return self_locs

def compare_two_attentions(seq_length,
		runLSH=True, runLSH_v2=False,
		runFull=False, runFull_v2=False):
	shape = batch_size, length, num_heads, num_dim = 4, seq_length, 2, 256
	bucket_count = 128
	print('>>>1. Prepare inputs(QK, V) with shape: ', shape)
	tf.random.set_seed(0)
	qk = tf.random.normal(shape)
	qk_norm = make_unit_length(qk)
	#qk_norm = tf.math.l2_normalize(qk, -1)
	v = tf.random.normal(shape)

	if runLSH:
		print('\n>>>2.1. run LSH attention')
		qk2 = tf.reshape(tf.transpose(qk, perm=[0,2,1,3]), (-1, length, num_dim))
		v2 = tf.reshape(tf.transpose(v, perm=[0,2,1,3]), (-1, length, num_dim))
		ret21 = calculate_LSH_attention(qk2, v2, None, 0, 2, length // bucket_count)
		ret21 = tf.transpose(tf.reshape(ret21, (batch_size, num_heads, length, num_dim)), perm=[0,2,1,3])
		print('ret21.shape: ', ret21.shape)
		#logits2 = tf.reshape(logits2, (batch_size, num_heads, length, -1))
		#print('logits1.shape: ', logits2.shape)

	if runLSH_v2:
		print('\n>>>2.1. run LSH attention')
		qk2 = tf.reshape(tf.transpose(qk, perm=[0,2,1,3]), (-1, length, num_dim))
		v2 = tf.reshape(tf.transpose(v, perm=[0,2,1,3]), (-1, length, num_dim))
		ret21 = calculate_LSH_attention_v2(qk2, v2, 0, False, 0., 2, length // bucket_count)
		ret21 = tf.transpose(tf.reshape(ret21, (batch_size, num_heads, length, num_dim)), perm=[0,2,1,3])

	if runFull:
		print('\n>>>3.1. run full attention')
		self_bias = get_self_attention_mask(length) * (-1e5)
		print('self_bias: ', self_bias.shape)

		q3 = qk * (num_dim ** -0.5)
		ret31 = calculate_full_attention(qk_norm, q3, v, self_bias, False, 0)

	if runFull_v2:
		print('\n>>>3.2. run full attention v2')
		#self_bias = get_self_attention_mask(length) * (-1e5)
		#print('self_mask: ', self_mask.shape)

		ret32 = calculate_full_attention_v2(qk, qk_norm, v, None, apply_soft_selfmask=True)
		#logits2 = tf.reshape(logits2, (-1, length, length))
		#print('logits2.shape: ', logits2.shape)

	if runLSH and runFull:
		print('\n>>>4.1. compare results')
		#loss = tf.keras.losses.MeanAbsoluteError()(ret1, ret2)
		#loss2 = tf.keras.losses.MeanSquaredError()(ret1, ret2)
		#print(f'loss = {loss}, loss2 = {loss2}')
		print('logits1: ', logits1[0])
		print('logits2: ', logits2[0])
		#sumexp1 = tf.reduce_sum(tf.exp(logits1), axis=-1)
		#sumexp2 = tf.reduce_sum(tf.exp(logits2), axis=-1)
		#print(sumexp1[0])
		#print(sumexp2[0])
		#print(tf.sort(logits1[0]))
		#print(tf.sort(logits2[0]))
		#mean_percent = tf.reduce_mean(tf.exp(logits1 - logits2))
		#print(f'mean_percent = {mean_percent}')

	if runFull and runFull_v2:
		print('\n>>>4.2. compare results')
		print('ret31: ', ret31[0,1,0,:64])
		print('ret32: ', ret32[0,1,0,:64])
		diffcount = tf.reduce_sum(tf.cast(tf.abs(ret31-ret32) > 1e-9, tf.int32))
		print(f'diff count: {diffcount}')

#compare_two_attentions(1024 * 8)

#def check_two_attentions_memory_usage(checkLSH=True):
#	attention_type = 'LSH' if checkLSH else 'full'
#	for i in range(1, 64):
#		print(f'checking {attention_type} attention when seq_length = {i}K ...')
#		compare_two_attentions(1024*i, runLSH=checkLSH, runFull=not checkLSH)
#		print(f'OK: {attention_type} attention when seq_length = {i}K.')
#
#check_two_attentions_memory_usage(checkLSH=True) # max length 35K
#check_two_attentions_memory_usage(checkLSH=False) # max length 8K
