#!/usr/bin/env python3
"""data pre-processing for Gigaword

convert_targz2articles(): parse tar.gz file and print (id, summary, article)
"""

import sys
import os
import gzip

sys.path.append('..')
import data_generic as dg

def _fix_paren(parse):
	"""fix some parses are mis-parenthesized."""
	if len(parse) < 2:
		return parse
	if parse[0] == "(" and parse[1] == " ":
		return parse[2:-1]
	return parse

def _get_words(parse):
	words = []
	for w in parse.split():
		if w[-1] == ')' and len(w) > 1:
			words.append(w.strip(")"))
			#if words[-1] == ".": break
	return words

def _is_good(_title_words, _article):
	_article_words = [w for sent in _article for w in sent]

	if not '.' in [w for sent in _article[:2] for w in sent]:
		return False

	# spurious words to blacklist for titles.
	# 1. first set is words that never appear in input and output
	# 2. second set is punctuation and non-title words.
	stop_words = set(['update#', 'update', 'recasts', 'undated', 'grafs', 'corrects',
				'retransmitting', 'updates', 'dateline', 'writethru',
				'recaps', 'inserts', 'incorporates', 'adv#', 'adv##',
				'ld-writethru', 'djlfx', 'edits', 'byline',
				'repetition', 'background', 'thruout', 'quotes',
				'attention', 'ny#', 'ny###', 'overline', 'embargoed', 'ap', 'gmt',
				'edt', 'adds', 'embargo',
				'urgent', '?', 'i', ':', '-', 'by', '-lrb-', '-rrb-', 'afp'])
	if any([w.lower() in stop_words for w in _title_words]) > 0:
	   return False

	# reasonable lengths
	if not (10 < len(_article_words) <= 2000 and
			3 < len(_title_words) < 100):
		return False

	# some word match.
	matches = len(set([w.lower() for w in _title_words if len(w) > 3]) &
				set([w.lower() for w in _article_words[:128] if len(w) > 3]))
	if matches < 1:
		return False

	return True

def convert_targz2articles(inputfn):
	MAX_ARTICLE_SENTENCE_COUNT = 2
	NONE, HEAD, NEXT, TEXT = 0, 1, 2, 3

	MODE = NONE
	doc_id = ''
	title = []
	article = []

	#line_count = 0
	for l in gzip.open(inputfn, 'rt'):
		#line_count += 1
		#if line_count > 10000: break

		ls = l.strip()

		if ls.find('<DOC id="') == 0:
			doc_id = ls.split('"')[1]
		elif MODE == NONE and ls.strip() == "<HEADLINE>":
			MODE = HEAD
		elif MODE == HEAD:
			title = _get_words(_fix_paren(ls))
			MODE = NEXT
		elif MODE == NEXT and ls == "<TEXT>":
			MODE = TEXT
		elif MODE == TEXT and ls == "</TEXT>":
			#if "(. .)" not in article_parse[0]:
			#	print(line_count, article_parse[0])

			if _is_good(title, article):
				# schema: doc_it \t summary \t article
				sentences = [' '.join(words) for words in article]
				print('\t'.join([
					doc_id,
					' '.join(title),
					dg.TOKEN_EOS_SPACES.join(sentences[:MAX_ARTICLE_SENTENCE_COUNT])
					]))

			doc_id = ''
			title = []
			article = []
			MODE = NONE
		elif MODE == TEXT:
			if ls == "<P>" or ls == "</P>": continue
			sentence_words = _get_words(_fix_paren(ls))
			if len(sentence_words) > 0:
				article.append(sentence_words)


if __name__ == '__main__':
	convert_targz2articles(sys.argv[1])

