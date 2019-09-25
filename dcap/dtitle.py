import sys
import os.path
import time

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import argparse

import transformer_model as tsfm

parser = argparse.ArgumentParser(description='train or evaluate deep summarization models')
parser.add_argument('--mode', choices=['train', 'eval', 'check-data'], help='execuation mode: e.g. train, eval', required=True)
parser.add_argument('--vocab', default='vocab', help='the vocab file for SubwordTextEncoder')
parser.add_argument('--training_data', default='cap_title/training.titles')
parser.add_argument('--test_data', default='cap_title/test.titles')
parser.add_argument('--model_path', default='model')
parser.add_argument('--batch_size', type=int, default=32, help='the mini-batch size for training')
parser.add_argument('--shuffle_buffer_size', type=int, default=8192)
parser.add_argument('--max_body_length', type=int, default=1024)
parser.add_argument('--max_title_length', type=int, default=48)
FLAGS, _ = parser.parse_known_args()
print('FLAGS = {}'.format(FLAGS))


train_examples = tf.data.TextLineDataset(FLAGS.training_data)
train_examples = train_examples.take(512*1024)
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
		body.numpy())[:FLAGS.max_body_length-2] + [tokenizer.vocab_size+1]
	return title, body

def tf_encode(item):
	return tf.py_function(encode, [item[0], item[1]], [tf.int64, tf.int64])

def filter_max_length(title, body):
	#return tf.logical_and(tf.size(body) <= FLAGS.max_body_length, tf.size(title) <= FLAGS.max_title_length)
	return tf.size(title) <= FLAGS.max_title_length

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
#train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(FLAGS.shuffle_buffer_size)
train_dataset = train_dataset.padded_batch(FLAGS.batch_size, padded_shapes=([FLAGS.max_title_length], [FLAGS.max_body_length]), drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


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
  
	return enc_padding_mask, combined_mask

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

vocab_size = tokenizer.vocab_size + 2
dropout_rate = 0.1

def train_model():
	transformer = tsfm.Transformer(num_layers, d_model, num_heads, dff,
	                          vocab_size, FLAGS.max_body_length, FLAGS.max_title_length, dropout_rate)

	learning_rate = tsfm.CustomSchedule(d_model)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

	ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.model_path, max_to_keep=50)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print('Latest checkpoint restored from {}'.format(ckpt_manager.latest_checkpoint))

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

	# The @tf.function trace-compiles train_step into a TF graph for faster
	# execution. The function specializes to the precise shape of the argument
	# tensors. To avoid re-tracing due to the variable sequence lengths or variable
	# batch sizes (the last batch is smaller), use input_signature to specify
	# more generic shapes.
	train_step_signature = [
		tf.TensorSpec(shape=(FLAGS.batch_size, None), dtype=tf.int64),
		tf.TensorSpec(shape=(FLAGS.batch_size, None), dtype=tf.int64),
	]

	@tf.function(input_signature=train_step_signature)
	def train_step(inp, tar):
		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
  
		enc_padding_mask, look_ahead_mask = create_masks(inp, tar_inp)
  
		with tf.GradientTape() as tape:
			predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, look_ahead_mask)
			loss = loss_function(tar_real, predictions, loss_object)

		gradients = tape.gradient(loss, transformer.trainable_variables)    
		optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
		train_loss(loss)
		train_accuracy(tar_real, predictions, sample_weight=tf.math.not_equal(tar_real, 0))
#		tf.print(tar_real, summarize=100)
#		tf.print(tf.argmax(predictions, axis=-1), summarize=100)

	def save_checkpoint(epoch, batch):
		ckpt_save_path = ckpt_manager.save()
		print('Saving checkpoint for epoch/batch {}/{} at {}'.format(epoch, batch, ckpt_save_path))
		print('Current Loss {:.4f} Accuracy {:.4f}.'.format(train_loss.result(), train_accuracy.result()))
		train_loss.reset_states()
		train_accuracy.reset_states()

	EPOCHS = 50
	print('Start of model training for {} epoch(es) at {}'.format(EPOCHS, time.asctime()))
	for epoch in range(EPOCHS):
		start = time.time()
		train_loss.reset_states()
		train_accuracy.reset_states()
  
		# inp -> html body, tar -> html title
		for (batch, (tar, inp)) in enumerate(train_dataset):
			train_step(inp, tar)
			#print('{}: {} / {}'.format(batch, inp[0][:10], tar[0][:10]))
			if batch % 100 == 0 or batch < 5:
				print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} at {}'.format(
				       epoch, batch, train_loss.result(), train_accuracy.result(), time.asctime()))
			if batch % 10000 == 0 and batch > 0:
				save_checkpoint(epoch, batch)

		save_checkpoint(epoch, batch)
		print('Time taken for epoch#{}: {} secs\n'.format(epoch, time.time() - start))


def eval_model():
	transformer = tsfm.Transformer(num_layers, d_model, num_heads, dff,
	                          input_vocab_size, target_vocab_size, dropout_rate)

	ckpt = tf.train.Checkpoint(model=transformer)
	ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.model_path, max_to_keep=None)

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


def check_data():
	print('\n*** check test data')
	test_batch_size = 2
	test_examples = tf.data.TextLineDataset(FLAGS.test_data)
	test_examples = test_examples.map(lambda ln: tf.strings.split(ln, '\t')[1:])
	test_dataset = test_examples.map(tf_encode)
	test_dataset = test_dataset.filter(filter_max_length).padded_batch(test_batch_size, padded_shapes=([-1], [-1]))
	res = next(iter(test_dataset))
	print('first {} samples in test data:\n{}'.format(test_batch_size, res))
	for i in range(test_batch_size):
		tokens = [i for i in res[0][i].numpy() if i < tokenizer.vocab_size and i > 0]
		print('decoded example #{}:\n\t[{}]'.format(i, tokenizer.decode(tokens)))
		print('\t' + '/'.join((tokenizer.decode([i]) for i in tokens if i < tokenizer.vocab_size and i > 0)))

	print('\n*** check training data')
	titles, bodies = [], []
	lc, msg_step, dot_per_msg = 0, 100000, 40
	for l in open(FLAGS.training_data, 'r', encoding='utf8'):
		inputs = l.strip().split('\t')
		title, body = encode(tf.constant(inputs[1]), tf.constant(inputs[2]))
		titles.append(len(title))
		bodies.append(len(body))
		lc += 1
		if lc % (msg_step//dot_per_msg) == 0: print('.', end='', flush=True)
		if lc % msg_step == 0: print(' load {}k training examples'.format(lc//1000))
		if lc == 400000: break
	print('histogram of {} training examples:'.format(len(titles)))
	print('title''s histogram: {}'.format(list(zip(*np.histogram(titles, bins = range(FLAGS.max_title_length+1), density=True)))))
	print('body''s histogram: {}'.format(list(zip(*np.histogram(bodies, bins = range(0, FLAGS.max_body_length+10, 10), density=True)))))


if FLAGS.mode == 'train':
	train_model()
elif FLAGS.mode == 'eval':
	pass #eval_model()
elif FLAGS.mode == 'check-data':
	check_data()

