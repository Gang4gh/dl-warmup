#!/usr/bin/env python3

import sys
import re
import collections

import numpy as np


# special tokens
TOKEN_PAD = '<PAD>'
TOKEN_OOV = '<OOV>'
TOKEN_START = '<START>'
TOKEN_END = '<END>'
TOKEN_EOS = '<eos/>'
TOKEN_EOS_SPACES = ' <eos/> '	# not in vocabulary


def _normalize_text_v1(text):
	return text.lower()

def _normalize_text_v2(text):
	return re.sub(r'\d(?:[,./\\\d]*\d)?', '#', text.lower())


_stop_ngrams = set(['major news items ', 'weather forecast for ', 'cox news service ', 'beijing-based newspapers'
		, ', am editors', ' economic briefs', ' business briefs', ' key market information'
		])


_FilterConfig = collections.namedtuple('Config',
				'name normalize_fn max_summary_sentence max_article_sentence '
				'max_summary_words max_article_words truncate_when_exceed ')
_config_Gigaword = _FilterConfig(name = 'Gigaword_config',
				normalize_fn = _normalize_text_v2,
				max_summary_sentence=9999, max_article_sentence=2,
				max_summary_words=30, max_article_words=120,
				truncate_when_exceed=False)
_config_CnnDM = _FilterConfig(name = 'CNN/DailyMail_config',
				normalize_fn = _normalize_text_v1,
				max_summary_sentence=9999, max_article_sentence=9999,
				max_summary_words=100, max_article_words=400,
				truncate_when_exceed=True)


def filter_articles(filefn, data_source):
	config = _config_Gigaword if data_source == 'Gigaword' else _config_CnnDM

	f = sys.stdin if filefn == '-' else open(filefn)
	for l in f:
		splits = l.strip().split('\t')
		if len(splits) != 3:
			continue

		doc_id, summary, article = splits[0], config.normalize_fn(splits[1]), config.normalize_fn(splits[2])
		summary_words = [w for sent in summary.split(TOKEN_EOS_SPACES)[:config.max_summary_sentence] if len(sent) > 0 for w in sent.split() + [TOKEN_EOS]][:-1]
		article_words = [w for sent in article.split(TOKEN_EOS_SPACES)[:config.max_article_sentence] if len(sent) > 0 for w in sent.split() + [TOKEN_EOS]][:-1]

		# reasonable lengths
		if config.truncate_when_exceed:
			summary_words = summary_words[:config.max_summary_words]
			article_words = article_words[:config.max_article_words]
		else:
			if not (10 < len(article_words) <= config.max_article_words and
					3 < len(summary_words) < config.max_summary_words):
				continue

		summary = ' '.join(summary_words)
		article = ' '.join(article_words)

		# filter by ngrams stop words
		if any([ngram in summary for ngram in _stop_ngrams]):
			continue

		print('\t'.join([doc_id, summary, article]))


def build_vocab(filefn, max_allowed_freq):
	counter = collections.Counter()

	for l in open(filefn):
		splits = l.strip().split("\t")
		if len(splits) != 3:
			continue
		_, title, article = splits
		counter.update(title.split())
		counter.update(article.split())

	del counter[TOKEN_EOS]

	for word, count in counter.most_common():
		if count < max_allowed_freq:
			break
		print(word, count)


def count_titles(filefn, max_allowed_count):
	counter = collections.Counter()

	for l in open(filefn):
		splits = l.strip().split("\t")
		if len(splits) != 3:
			continue
		_, title, _ = splits
		counter.update([title])

	for p in counter.most_common():
		if p[1] == max_allowed_count: break
		print(p)


def calc_histogram(filefn):
	data = ({}, {})

	for l in open(filefn):
		splits = l.strip().split("\t")
		if len(splits) != 3:
			continue
		id, title, article = splits
		id = id[:3]
		if id not in data[0]:
			data[0][id] = []
			data[1][id] = []
		data[0][id].append(len(title.split()))
		data[1][id].append(len(article.split()))

	for key in data[0]:
		print(key)
		print('	median word count in title :', np.median(data[0][key]))
		print(np.histogram(data[0][key], density=True))
		print('	median word count in article :', np.median(data[1][key]))
		print(np.histogram(data[1][key], density=True))


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('error: invalid argv.', sys.argv)
		exit(-1)

	cmd = sys.argv[1]
	filefn = sys.argv[2] if len(sys.argv) > 2 else None

	if cmd == 'build-vocab':
		build_vocab(filefn, 10)
	elif cmd == 'filter-articles':
		filter_articles(filefn, sys.argv[3])
	elif cmd == 'count-title':
		count_titles(filefn, 1)
	elif cmd == 'histogram':
		calc_histogram(filefn)
	else:
		print('error: invalid command.', cmd)
		exit(-1)
