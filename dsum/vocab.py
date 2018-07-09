"""Vocab class to convert word to id and vice versa"""

import glob
import random
import logging

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
		self.token_eos_id = self._add_word(dg.TOKEN_EOS)
		SPECIAL_TOKEN_COUNT = len(self._word_to_id)

		with open(vocab_file_path) as f:
			for line in f:
				parts = line.split()
				if len(parts) != 2:
					raise ValueError('Invalid vocab file, line = [%s]' % line)
				self._add_word(parts[0])
				if len(self._word_to_id) >= vocab_size_cap + SPECIAL_TOKEN_COUNT: # special tokens added before
					break

		self._static_vocab_size = len(self._word_to_id)
		logging.info('load %d tokens (including %d special tokens, %d index tokens) into vocab.',
			self.get_vocab_size(), SPECIAL_TOKEN_COUNT, self._index_vocab_size)

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
					logging.warning('rid(%d) is out of index_vocab_size(%d)', rid, self._index_vocab_size)
					return self.token_oov_id
				else:
					return self._static_vocab_size + rid
			except ValueError:
				return self.token_oov_id

	def get_word_by_id(self, id, reference=None, markup=False):
		if id >= 0 and id < self._static_vocab_size:
			return self._id_to_word[id]
		elif id >= self._static_vocab_size and id < self.get_vocab_size():
			rid = id - self._static_vocab_size
			word = reference[rid] if rid < len(reference) else dg.TOKEN_OOV
			if markup:
				return '__%s_(%d/%d)' % (word, rid, self.get_id_by_word(word))
			else:
				return word
		else:
			raise ValueError('Invalid vocab id: %s' % id)

	def parse_article(self, article_line):
		_, summary, article = article_line.split('\t')
		article_tokens = [word for word in article.split()]
		summary_tokens = [word for word in summary.split()]
		article_ids = [self.get_id_by_word(word, article_tokens) for word in article_tokens]
		summary_ids = [self.get_id_by_word(word, article_tokens) for word in summary_tokens]
		return article_ids, article_tokens, article, summary_ids, summary_tokens, summary


def check_vocab_stats(article_path, vocab_path, vocab_cap = 50000, vocab_index_size = 120):
	vocab = Vocab(vocab_path, vocab_cap, vocab_index_size)
	line_count, total_word_count, index_word_count, oov_word_count = 0, 0, 0, 0
	for line in open(article_path):
		art_ids, _, _, summary_ids, _, _ = vocab.parse_article(line.strip())
		line_count += 1
		total_word_count += len(summary_ids)
		index_word_count += sum(1 for wid in summary_ids if wid >= vocab_cap + 4)
		oov_word_count += sum(1 for wid in summary_ids if wid == vocab.token_oov_id)
	print('read %d lines' % line_count)
	print('total_word_count = %d, index_word_count = %d, oov_word_count = %d' % (total_word_count, index_word_count, oov_word_count))
	print('index/total = %f, oov/total = %f' % (index_word_count / total_word_count, oov_word_count / total_word_count))

#check_vocab_stats('training.articles', 'training.vocab', 10000)
