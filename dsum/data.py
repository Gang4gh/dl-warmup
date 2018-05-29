"""Data vocab and reader for summerization data in <tab> separated text files"""

import sys
import glob
import random
import struct

# Special tokens
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

EOS_TOKEN = '<eos/>'				# End of sentence token, used as sentence boundary
EOS_TOKEN2 = ' ' + EOS_TOKEN + ' '

class Vocab(object):
	"""Vocabulary class to map words and ids."""

	def __init__(self, vocab_file, max_vocab_size):
		self._word_to_id = {}
		self._id_to_word = {}
		self._count = 0

		with open(vocab_file, 'r') as vocab_f:
			for line in vocab_f:
				pieces = line.split()
				if len(pieces) != 2:
					sys.stderr.write('Bad line: %s\n' % line)
					continue
				if pieces[0] in self._word_to_id:
					raise ValueError('Duplicated word: %s.' % pieces[0])
				self._word_to_id[pieces[0]] = self._count
				self._id_to_word[self._count] = pieces[0]
				self._count += 1
				if self._count >= max_vocab_size:
					print('Too many words: >%d.' % max_vocab_size)
					break
			for i in range(120):
				word = '[#%d]' % i
				self._word_to_id[word] = self._count
				self._id_to_word[self._count] = word
				self._count += 1
		
		# Check for presence of required special tokens.
		assert self.CheckVocab(PAD_TOKEN) > 0
		assert self.CheckVocab(UNKNOWN_TOKEN) >= 0
		assert self.CheckVocab(SENTENCE_START) > 0
		assert self.CheckVocab(SENTENCE_END) > 0

		print('load vocab done with {} tokens.'.format(self._count))

	def CheckVocab(self, word):
		if word not in self._word_to_id:
			return None
		return self._word_to_id[word]

	def WordToId(self, word):
		if word not in self._word_to_id:
			return self._word_to_id[UNKNOWN_TOKEN]
		return self._word_to_id[word]

	def IdToWord(self, word_id):
		if word_id not in self._id_to_word:
			raise ValueError('id not found in vocab: %d.' % word_id)
		return self._id_to_word[word_id]

	def NumIds(self):
		return self._count


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


def Pad(ids, pad_id, length):
	"""Pad or trim list to len length.

	Args:
		ids: list of ints to pad
		pad_id: what to pad with
		length: length to pad or trim to

	Returns:
		ids trimmed or padded with pad_id
	"""
	assert pad_id is not None
	assert length is not None

	if len(ids) < length:
		a = [pad_id] * (length - len(ids))
		return ids + a
	else:
		return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
	"""Get ids corresponding to words in text.

	Assumes tokens separated by space.

	Args:
		text: a string
		vocab: TextVocabularyFile object
		pad_len: int, length to pad to
		pad_id: int, word id for pad symbol

	Returns:
		A list of ints representing word ids.
	"""
	ids = []
	for w in text.split():
		i = vocab.WordToId(w)
		if i >= 0:
			ids.append(i)
		else:
			ids.append(vocab.WordToId(UNKNOWN_TOKEN))
	if pad_len is not None:
		return Pad(ids, pad_id, pad_len)
	return ids

def DoGetWordId(word, vocab, reference):
	wid = vocab.CheckVocab(word)
	if wid is not None:
		return wid
	try:
		rid = reference.index(word)
		return vocab.WordToId('[#%d]' % rid)
	except ValueError:
		return vocab.WordToId(UNKNOWN_TOKEN)

def GetWordIds2(article, abstract, vocab):
	article_words = [w for sent in article for w in sent.split()]
	article_word_ids = [DoGetWordId(w, vocab, article_words) for w in article_words]
	abstract_word_ids = [DoGetWordId(w, vocab, article_words) for sent in abstract for w in sent.split()]
	return (article_word_ids, abstract_word_ids)


def Ids2Words(ids_list, vocab):
	"""Get words from ids.

	Args:
		ids_list: list of int32
		vocab: TextVocabulary object

	Returns:
		List of words corresponding to ids.
	"""
	assert isinstance(ids_list, list), '%s  is not a list' % ids_list
	return [vocab.IdToWord(i) for i in ids_list]

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
