import sys
import os.path
import time

import tensorflow_datasets as tfds
import tensorflow as tf
#import numpy as np
import argparse

import transformer_model as tsfm

parser = argparse.ArgumentParser(description='train or evaluate deep summarization models')
parser.add_argument('--mode', choices=['train', 'eval', 'test'], help='train / eval mode', required=True)
parser.add_argument('--vocab', default='vocab', help='the vocab file for SubwordTextEncoder')
parser.add_argument('--batch_size', type=int, default=64, help='the mini-batch size for training')
parser.add_argument('--shuffle_buffer_size', type=int, default=8192)
parser.add_argument('--max_body_length', type=int, default=2048)
parser.add_argument('--max_title_length', type=int, default=50)
FLAGS, _ = parser.parse_known_args()
print('FLAGS = {}'.format(FLAGS))

# load data and prepare dictionaries if necessary
#examples, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True, shuffle_files=False)
#val_examples = examples['validation']
#print(next(iter(val_examples)))
#sys.exit(0)

train_examples = tf.data.TextLineDataset('cap_title/training.titles')
train_examples = train_examples.map(lambda ln: tf.strings.split(ln, '\t')[1:])

# load en/pt tokenizers
if os.path.isfile(FLAGS.vocab + '.subwords'):
	tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(FLAGS.vocab)
	print('load tokenizer from "{}"'.format(FLAGS.vocab))
else:
	print('{}: start of building vocab.'.format(time.asctime()))
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
		(s.numpy() for tb in train_examples for s in tb),
		target_vocab_size=2**13)
	tokenizer.save_to_file(FLAGS.vocab)
	print('{}: prepare tokenizers and save to "{}".'.format(time.asctime(), FLAGS.vocab))

def encode(title, body):
	title = [tokenizer.vocab_size] + tokenizer.encode(
		title.numpy()) + [tokenizer.vocab_size+1]
	body = [tokenizer.vocab_size] + tokenizer.encode(
		body.numpy()) + [tokenizer.vocab_size+1]
	return title, body

def tf_encode(item):
	return tf.py_function(encode, [item[0], item[1]], [tf.int64, tf.int64])

def filter_max_length(title, body):
	return tf.logical_and(tf.size(body) <= FLAGS.max_body_length, tf.size(title) <= FLAGS.max_title_length)

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
#train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(FLAGS.shuffle_buffer_size)
train_dataset = train_dataset.padded_batch(FLAGS.batch_size, padded_shapes=([-1], [-1]), drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#for _ in range(3):
#	print('\n{}: start of counting'.format(time.asctime()))
#	item_count, start_time = 0, time.time()
#	for (batch, (tar, inp)) in enumerate(train_dataset.take(501)):
#		if batch == 0: start_time = time.time()
#		if batch % 100 == 0: print('\tbatch = {} at {}'.format(batch, time.asctime()))
#		item_count += FLAGS.batch_size
#	print('item_count = {}, time_cost = {}'.format(item_count, time.time() - start_time))
#	print('{}: end of counting'.format(time.asctime()))
#sys.exit(0)

#val_dataset = val_examples.map(tf_encode)
#val_dataset = val_dataset.filter(filter_max_length).padded_batch(FLAGS.batch_size, padded_shapes=([-1], [-1]))
#pt_batch, en_batch = next(iter(val_dataset))
#print(pt_batch, en_batch)

def loss_function(real, pred, loss_object):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
  
	return tf.reduce_mean(loss_)

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
	# Encoder padding mask
	enc_padding_mask = create_padding_mask(inp)
  
	# Used in the 2nd attention block in the decoder.
	# This padding mask is used to mask the encoder outputs.
	dec_padding_mask = create_padding_mask(inp)
  
	# Used in the 1st attention block in the decoder.
	# It is used to pad and mask future tokens in the input received by 
	# the decoder.
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
	return enc_padding_mask, combined_mask, dec_padding_mask

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

vocab_size = tokenizer.vocab_size + 2
dropout_rate = 0.1

checkpoint_path = "./model/train"
learning_rate = tsfm.CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def train_model():
	transformer = tsfm.Transformer(num_layers, d_model, num_heads, dff,
	                          vocab_size, FLAGS.max_body_length, FLAGS.max_title_length, dropout_rate)

	ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print('Latest checkpoint restored')

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	# The @tf.function trace-compiles train_step into a TF graph for faster
	# execution. The function specializes to the precise shape of the argument
	# tensors. To avoid re-tracing due to the variable sequence lengths or variable
	# batch sizes (the last batch is smaller), use input_signature to specify
	# more generic shapes.
	train_step_signature = [
		tf.TensorSpec(shape=(None, None), dtype=tf.int64),
		tf.TensorSpec(shape=(None, None), dtype=tf.int64),
	]

	@tf.function(input_signature=train_step_signature)
	def train_step(inp, tar):
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
  
		enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
		with tf.GradientTape() as tape:
			predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
			loss = loss_function(tar_real, predictions, loss_object)

		gradients = tape.gradient(loss, transformer.trainable_variables)    
		optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
		train_loss(loss)
		train_accuracy(tar_real, predictions)

	EPOCHS = 50
	print('Start of model training for {} epoch(es) at {}'.format(EPOCHS, time.asctime()))
	for epoch in range(EPOCHS):
		start = time.time()
		train_loss.reset_states()
		train_accuracy.reset_states()
  
		# inp -> portuguese, tar -> english
		for (batch, (tar, inp)) in enumerate(train_dataset):
			train_step(inp, tar)
			#print('{}: {} / {}'.format(batch, inp[0][:10], tar[0][:10]))
			if batch % 50 == 0 or batch <= 5:
				print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} at {}'.format(
				       epoch + 1, batch, train_loss.result(), train_accuracy.result(), time.asctime()))
			if batch % 1000 == 0:
				ckpt_save_path = ckpt_manager.save()
				print('Saving checkpoint for epoch/batch {}/{} at {}'.format(epoch+1, batch, ckpt_save_path))

		ckpt_save_path = ckpt_manager.save()
		print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
		print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
		print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def eval_model():
	transformer = tsfm.Transformer(num_layers, d_model, num_heads, dff,
	                          input_vocab_size, target_vocab_size, dropout_rate)

	ckpt = tf.train.Checkpoint(model=transformer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=None)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
		print('Latest checkpoint restored')
	else:
		print('Failed to restore model')
		sys.exit(-1)

	translate("este é um problema que temos que resolver.",
              "This is a problem we have to solve .", transformer)
	translate("Boa vinda à porcelana, todos",
              "Welcome to China, everybody.", transformer)

def translate(sentence, tar_sentence, transformer):
	start_token = [tokenizer_pt.vocab_size]
	end_token = [tokenizer_pt.vocab_size + 1]

	# inp sentence is portuguese, hence adding the start and end token
	inp_sentence = start_token + tokenizer_pt.encode(sentence) + end_token
	encoder_input = tf.expand_dims(inp_sentence, 0)

	# as the target is english, the first word to the transformer should be the
	# english start token.
	decoder_input = [tokenizer_en.vocab_size]
	output = tf.expand_dims(decoder_input, 0)

	for i in range(FLAGS.max_title_length):
		enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

		# predictions.shape == (batch_size, seq_len, vocab_size)
		predictions, attention_weights = transformer(encoder_input, output, False,
                                                     enc_padding_mask, combined_mask, dec_padding_mask)

		# select the last word from the seq_len dimension
		predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

		# return the result if the predicted_id is equal to the end token
		if predicted_id == tokenizer_en.vocab_size+1:
			break
			#return tf.squeeze(output, axis=0), attention_weights

		# concatentate the predicted_id to the output which is given to the decoder
		# as its input.
		output = tf.concat([output, predicted_id], axis=-1)

	result, attention_weights = tf.squeeze(output, axis=0), attention_weights
	predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
	print('\nSentence  : {}'.format(sentence))
	print('Inp_sent  : {}'.format(inp_sentence))
	print('Prediction: {}'.format(predicted_sentence))
	print('Target    : {}'.format(tar_sentence))

if FLAGS.mode == 'train':
	train_model()
elif FLAGS.mode == 'eval':
	eval_model()

