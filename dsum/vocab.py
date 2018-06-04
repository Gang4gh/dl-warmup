"""Vocab class to convert word to id and vice versa"""

import glob
import random

import data_generic as dg


class Vocab():
	"""Vocabulary class that supports 'index pointer'"""
	def __init__(self, vocab_file_path, vocab_size_cap, index_vocab_size=0):
		""" Args:
				vocab_file_path: path of .vocab file with tokens ordered by frequencies
				vocab_size_cap: max vocab size including special tokens
				index_vocab_size: max input length as index vocab size, 0 to disable pointer
		"""
		self._word_to_id = {}
		self._id_to_word = {}
		self._index_vocab_size = index_vocab_size

		self.token_pad_id = self._add_word(dg.TOKEN_PAD)
		self.token_oov_id = self._add_word(dg.TOKEN_OOV)
		self.token_start_id = self._add_word(dg.TOKEN_START)
		self.token_end_id = self._add_word(dg.TOKEN_END)

		with open(vocab_file_path) as f:
			for line in f:
				parts = line.split()
				if len(parts) != 2:
					raise ValueError('Invalid vocab file, line = [%s]' % line)
				self._add_word(parts[0])
				if len(self._word_to_id) >= vocab_size_cap + 4: # 4 special tokens added before
					break

		self._static_vocab_size = len(self._word_to_id)
		print('load %d tokens (including 4 special tokens, %d index tokens) into vocab.' %
			(self.get_vocab_size(), self._index_vocab_size))

	def _add_word(self, word):
		if word in self._word_to_id:
			raise ValueError('Duplicated word: %s' % word)
		word_id = len(self._word_to_id)
		self._word_to_id[word] = word_id
		self._id_to_word[word_id] = word
		return word_id

	def get_vocab_size(self):
		return self._static_vocab_size + self._index_vocab_size

	def get_id_by_word(self, word, reference=None):
		if word in self._word_to_id:
			return self._word_to_id[word]
		elif self._index_vocab_size == 0 or reference is None:
			return self.token_oov_id
		else:
			try:
				rid = reference.index(word)
				if rid >= self._index_vocab_size:
					raise IndexError('Invalid index token id, rid = %d' % rid)
				return self._static_vocab_size + rid
			except ValueError:
				return self.token_oov_id

	def get_word_by_id(self, id, reference=None, markup=False):
		if id >= 0 and id < self._static_vocab_size:
			return self._id_to_word[id]
		elif id >= self._static_vocab_size and id < self.get_vocab_size():
			rid = id - self._static_vocab_size
			if markup:
				return '__%s_(%d/%d)' % (reference[rid], rid, self.get_id_by_word(reference[rid]))
			else:
				return reference[rid] if rid < len(reference) else dg.TOKEN_OOV
		else:
			raise ValueError('Invalid vocab id: %s' % id)


def ExampleGen(data_path, num_epochs=None):
	"""Generates tf.Examples from path of data files.

		Binary data format: <length><blob>. <length> represents the byte size
		of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
		the tokenized article text and summary.

	Args:
		data_path: path to tf.Example data files.
		num_epochs: Number of times to go through the data. None means infinite.

	Yields:
		Deserialized tf.Example.

	If there are multiple files specified, they accessed in a random order.
	"""
	epoch = 0
	while num_epochs is None or epoch < num_epochs:
		filelist = glob.glob(data_path)
		assert filelist, 'Empty filelist.'
		random.shuffle(filelist)
		for f in filelist:
			for l in open(f):
				splits = l.strip().split("\t")
				if len(splits) != 3:
					continue
				_, title, article = splits
				yield (article, title)
		print('end of epoch#', epoch)
		epoch += 1


def ToSentences(paragraph, include_token=True):
	"""Takes tokens of a paragraph and returns list of sentences.

	Args:
		paragraph: string, text of paragraph
		include_token: Whether include the sentence separation tokens result.

	Returns:
		List of sentence strings.
	"""
	s_gen = paragraph.split(' <eos/> ')
	return [s for s in s_gen]
