#!/usr/bin/env python3
"""data pre-processing for CNN/Daily Mail"""

import os
import sys
import hashlib

sys.path.append('..')
import data_generic as dg

#dm_single_close_quote = u'\u2019' # unicode
#dm_double_close_quote = u'\u201d'
#END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ')']

def get_hash_value(url):
	h = hashlib.sha1()
	h.update(url.encode('utf-8'))
	return h.hexdigest()

def print_parsed_story_file(fn):
	with open(fn) as f:
		lines = [line.strip() for line in f if line.strip() != '']
		end_of_body = lines.index('@highlight')
		article = [line for line in  lines[:end_of_body] if len(line) > 0]
		#highlights = [line + ('' if line[-1] in END_TOKENS else ' .') for line in lines[end_of_body:] if line != '@highlight' and len(line) > 0]
		highlights = [line for line in lines[end_of_body:] if line != '@highlight' and len(line) > 0]
		if len(article) == 0:
			return
		print('\t'.join([
			os.path.split(fn)[1].split('.')[0],
			dg.TOKEN_EOS_SPACES.join(highlights),
			dg.TOKEN_EOS_SPACES.join(article),		# actually, EOS is paragraph boundary for article field
			]))

def parse_and_merge(url_file, story_folder):
	for url in open(url_file):
		hash = get_hash_value(url.strip())
		fn = os.path.join(story_folder, hash + '.story')
		print_parsed_story_file(fn)

if __name__ == '__main__':
	parse_and_merge(*sys.argv[1:])

