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
	# run multiple instances in one GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	tf.keras.backend.set_session(tf.Session(config=config))

	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

def prepare_dataset(vocab, batch_size, data):
	if len(data) % batch_size != 0:
		data = data[:-(len(data) % batch_size)]

	def words_to_ids(text, max_word_count):
		ids = [vocab.get_id_by_word(w) for w in text.split()[:max_word_count]]
		return [vocab.token_pad_id] * (max_word_count - len(ids)) + ids

	x0 = [words_to_ids(rec.query, 16) for rec in data]
	x1 = [words_to_ids(rec.snippet1, 100) for rec in data]
	x2 = [words_to_ids(rec.snippet2, 100) for rec in data]
	y = tf.keras.utils.to_categorical([rec.label+1 for rec in data], 3)
	return [x0, x1, x2], y

def build_and_train_model(batch_size, model_dir, training_set, validation_set):
	query_input = Input(shape=(16,), dtype='int32')
	snippet1_input = Input(shape=(100,), dtype='int32')
	snippet2_input = Input(shape=(100,), dtype='int32')

	embedding = Embedding(input_dim=50000, output_dim=128)
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
		tf.keras.callbacks.ModelCheckpoint(model_dir + '/cp.{epoch:02d}.ckpt', verbose=1),
		tf.keras.callbacks.ModelCheckpoint(model_dir + '/cp.best.ckpt', verbose=1, save_best_only=True),
		tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3),
		tf.keras.callbacks.TensorBoard(log_dir='{0}/tb'.format(model_dir)),
		]

	model.fit(*training_set, epochs=100, batch_size=batch_size, callbacks=callbacks, validation_data=validation_set)
	model.save(model_dir + '/model')

	return model

Config = collections.namedtuple('Config', 'data_path vocab_path model_dir batch_size')
cfg = Config(
	data_path = 'sbsdatacleaned20181008_flips_tokLower.tsv',
	vocab_path = 'trainingdata.vocab',
	model_dir = 'model_v1030',
	batch_size = 512,
	)

if __name__ == '__main__':
	config_environment(cfg.model_dir)
	training_data, test_data = load_data(cfg.data_path) # data is shuffled by default

	vocab = Vocab(cfg.vocab_path, 50000)
	val_offset = len(training_data) // 10 # split 10% training data as validation data
	random.Random(1016).shuffle(training_data)
	training_set = prepare_dataset(vocab, cfg.batch_size, training_data[:-val_offset])
	validation_set = prepare_dataset(vocab, cfg.batch_size, training_data[-val_offset:])
	test_set = prepare_dataset(vocab, cfg.batch_size, test_data)
	print('training/validation/test set sizes : %d/%d/%d' % (len(training_set[0][0]), len(validation_set[0][0]), len(test_set[0][0])))

	model = build_and_train_model(cfg.batch_size, cfg.model_dir, training_set, validation_set)
	model = tf.keras.models.load_model(cfg.model_dir + '/cp.best.ckpt')
	pred = np.argmax(model.predict(test_set[0], batch_size=cfg.batch_size), -1)

	acc = accuracy_score(np.argmax(test_set[1], -1), pred)
	print('accuracy on test_set :', acc)

