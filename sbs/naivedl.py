import sys
import os
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score

from vocab import Vocab
from sbs_data import load_data

def config_environment(model_dir):
	# to run multiple instances on the same GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	tf.keras.backend.set_session(tf.Session(config=config))

	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

def prepare_dataset(vocab, batch_size, data, swap_left_and_right=False):
	if len(data) % batch_size != 0:
		data = data[:-(len(data) % batch_size)]

	def words_to_ids(text, max_word_count):
		ids = [vocab.get_id_by_word(w) for w in text.split()[:max_word_count]]
		return [vocab.token_pad_id] * (max_word_count - len(ids)) + ids

	x0 = [words_to_ids(rec.query, 16) for rec in data]
	x1 = [words_to_ids(rec.snippet1, 100) for rec in data]
	x2 = [words_to_ids(rec.snippet2, 100) for rec in data]
	y = [rec.label+1 for rec in data]
	if swap_left_and_right:
		x0, x1, x2, y = x0 + x0, x1 + x2, x2 + x1, y + [2-val for val in y]
	return [x0, x1, x2], tf.keras.utils.to_categorical(y, 3)

def build_and_train_model(batch_size, model_dir, training_set, validation_set):
	query_input = Input(shape=(16,), dtype='int32')
	snippet1_input = Input(shape=(100,), dtype='int32')
	snippet2_input = Input(shape=(100,), dtype='int32')

	embedding = Embedding(input_dim=50000, output_dim=64)
	query_embedding = embedding(query_input)
	snippet1_embedding = embedding(snippet1_input)
	snippet2_embedding = embedding(snippet2_input)

	query_lstm = LSTM(128)
	snippet_lstm = LSTM(128)
	query_encoding = query_lstm(query_embedding)
	snippet1_encoding = snippet_lstm(snippet1_embedding)
	snippet2_encoding = snippet_lstm(snippet2_embedding)

	all_encoding = tf.keras.layers.concatenate([query_encoding, snippet1_encoding, snippet2_encoding])
	X = Dense(256, activation='relu')(all_encoding)
	X = Dense(256, activation='relu')(X)
	X = Dense(256, activation='relu')(X)
	y = Dense(3, activation='softmax')(X)

	model = tf.keras.models.Model(inputs=[query_input, snippet1_input, snippet2_input], outputs=[y])
	model.summary()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(model_dir + '/cp.best.model', save_best_only=True),
		tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2),
		tf.keras.callbacks.TensorBoard(log_dir='{0}/tb'.format(model_dir)),
		]

	model.fit(*training_set, epochs=100, batch_size=batch_size, callbacks=callbacks, validation_data=validation_set)
	model = tf.keras.models.load_model(model_dir + '/cp.best.model')
	return model

Config = collections.namedtuple('Config', 'vocab_path model_dir batch_size')
cfg = Config(
	vocab_path = 'trainingdata.vocab',
	model_dir = 'model_v1101',
	batch_size = 512,
	)

def train_then_predict(training_data, test_data):
	config_environment(cfg.model_dir)

	vocab = Vocab(cfg.vocab_path, 50000)
	val_offset = len(training_data) // 10 # split 10% training data as validation data
	training_set = prepare_dataset(vocab, cfg.batch_size, training_data[:-val_offset])
	validation_set = prepare_dataset(vocab, cfg.batch_size, training_data[-val_offset:])
	test_set = prepare_dataset(vocab, 1, test_data)
	print('training/validation/test set sizes : %d/%d/%d' % (len(training_set[0][0]), len(validation_set[0][0]), len(test_set[0][0])))

	model = build_and_train_model(cfg.batch_size, cfg.model_dir, training_set, validation_set)
	pred = np.argmax(model.predict(test_set[0], batch_size=cfg.batch_size), -1)
	return [val-1 for val in pred]

if __name__ == '__main__':
	training_data, test_data = load_data() # struct data members: query snippet1 snippet2 weight label

	pred = train_then_predict(training_data, test_data)

	acc = accuracy_score([rec.label for rec in test_data], pred)
	print('accuracy on test_set :', acc)
