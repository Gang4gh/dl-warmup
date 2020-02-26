#!/usr/bin/env python3

import sys
import re
import time
import gzip
import collections

from absl import app
from absl import flags

def dtitle_reader(dtitle_file, input_schema, log_per_n_step=None):
	column_names = input_schema.split(',')
	Row = collections.namedtuple('Row', column_names, rename=True)

	lcount = 0

	is_gz_file = dtitle_file.endswith('.gz')
	fin = gzip.open(dtitle_file) if is_gz_file else open(dtitle_file, encoding='utf8')

	for l in fin:
		inputs = l.decode('utf8') if is_gz_file else l
		inputs = inputs.split('\t')
		if len(inputs) != len(column_names):
			print('invalid input, len(inputs)@{}!={}, {}'.format(len(inputs), len(column_names), inputs[0][:200]), file=sys.stderr)
			continue
		row = Row(*inputs[:-1], inputs[-1].strip())
		yield row
		if log_per_n_step:
			lcount += 1
			if lcount % log_per_n_step == 0:
				print('read {}k examples from {} at {}'.format(lcount//1024, dtitle_file, time.asctime()), file=sys.stderr)
	if log_per_n_step:
		print('read {} examples from {} in total'.format(lcount, dtitle_file), file=sys.stderr)


def preprocess_raw_input(FLAGS):
	DTitle_Row = collections.namedtuple('DTitle_Row', FLAGS.dtitle_schema)

	total, valid, suppressed = 0, 0, 0
	for row in dtitle_reader(FLAGS.input_file, FLAGS.input_schema):
		total += 1
		url, title, hostname, html = row.url, row.title, row.hostname, row.html
		if not url or not title or not html: continue

		#title = re.sub(r'#N#|#R#|#TAB#', ' ', title)
		#hostname = re.sub(r'#N#|#R#|#TAB#', ' ', hostname)
		#html = re.sub(r'</html>.*', '</html>', html, flags=re.I)

		if len(html) > FLAGS.doc_length_limit:
			m = re.match(f'.{{,{FLAGS.doc_length_limit}}}(?=[^\\w&</])', html)
			if m:
				html = m.group(0)
			else:
				print('invalid input, no-match, {}'.format(url), file=sys.stderr)
				continue

		if FLAGS.remove_title:
			html = re.sub(r'<title.*?</title>', ' ', html, flags=re.I)

		if FLAGS.remove_head:
			html = re.sub(r'<head.*?</head>', ' ', html, flags=re.I)

		url, title, hostname, html = (re.sub(r'\s+', ' ', s).strip().lower() for s in [url, title, hostname, html])
		title_tokens = [w for w in re.split(r'\W+', title) if w]

		if FLAGS.check_enoughtokens and len(title_tokens) <= 1:
			continue

		if FLAGS.check_exactmatch and title not in html:
			continue

		if FLAGS.check_fuzzymatch and any(token not in html for token in title_tokens):
			continue

		if (FLAGS.suppress_enoughtokens and len(title_tokens) <= 1
			or FLAGS.suppress_exactmatch and title not in html
			or FLAGS.suppress_fuzzymatch and any(token not in html for token in title_tokens)):
			if  FLAGS.max_suppress_ratio * (valid + suppressed) >= suppressed:
				suppressed += 1
				title = ''
			else:
				continue

		valid += 1 if title else 0
		if FLAGS.output_raw_html: html = row.html.strip().lower()
		print('\t'.join(DTitle_Row(url=url, title=title, hostname=hostname, html=html)))
	print('process {} examples, including {} ({:.2f}%) valid and {} ({:.2f}%) suppressed, from {}'.format(total, valid, valid/total*100, suppressed, suppressed/total*100, FLAGS.input_file), file=sys.stderr)


def build_vocab(FLAGS):
	import tensorflow_datasets as tfds
	target_vocab_file = '{}-{}'.format(FLAGS.vocab_file_prefix, FLAGS.target_vocab_size)
	print('{}: start to build a subwords tokenizer({}) with max_subword_length={}, max_corpus_chars={}GB.'.format(time.asctime(), target_vocab_file, FLAGS.max_subword_length, FLAGS.max_corpus_chars))
	corpus = (s.encode() for row in dtitle_reader(FLAGS.input_file, FLAGS.dtitle_schema, 100*1024) for s in [row.url, row.hostname, row.html, row.title])
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus,
			target_vocab_size = FLAGS.target_vocab_size,
			max_subword_length = FLAGS.max_subword_length,
			max_corpus_chars = int(FLAGS.max_corpus_chars*(2**30)),
			reserved_tokens = ['<EOS>', '<BOS#0>', '<BOS#1>', '<BOS#2>', '<BOS#3>'])
	tokenizer.save_to_file(target_vocab_file)
	print('{}: the subwords tokenizer({}) is ready.'.format(time.asctime(), target_vocab_file))


def tokenize_dtitle(FLAGS):
	import tensorflow as tf
	import tensorflow_datasets as tfds
	vocab_file = '{}-{}'.format(FLAGS.vocab_file_prefix, FLAGS.target_vocab_size)
	tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
	print(f'initilize tokenizer by vocab file [{vocab_file}].')

	assert FLAGS.input_file.endswith('.dtitle')
	tfrecord_file = FLAGS.input_file[:-7] + '.tfrecord'

	stats = []
	def _create_int64List_feature(text, limit):
		arr = tokenizer.encode(text)
		stats.append(len(arr))
		if limit:
			arr = arr[:limit]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))

	with tf.io.TFRecordWriter(tfrecord_file) as tfwriter:
		for row in dtitle_reader(FLAGS.input_file, FLAGS.dtitle_schema):
			datapoint = {
				'url': _create_int64List_feature(row.url, 0),
				'title': _create_int64List_feature(row.title, 0),
				'hostname': _create_int64List_feature(row.hostname, 0),
				'html': _create_int64List_feature(row.html, FLAGS.token_count_limit),
			}
			proto = tf.train.Example(features=tf.train.Features(feature=datapoint)).SerializeToString()
			tfwriter.write(proto)

	stats = sorted(stats[3::4])
	print(f'average token count = {sum(stats)/len(stats)}')
	print(f'75th percentile token count = {stats[len(stats) * 75 // 100]}')
	print(f'95th percentile token count = {stats[len(stats) * 95 // 100]}')
	print(f'99th percentile token count = {stats[len(stats) * 99 // 100]}')

	print(f'complete tokenization with token limit {FLAGS.token_count_limit}. write {len(stats)} outputs to {tfrecord_file}.')


def print_flags(FLAGS):
	print('FLAGS:')
	for f in FLAGS.get_key_flags_for_module(__file__):
		print('    {}: {}'.format(f.name, f.value))


def check_stats(FLAGS):
	import numpy as np
	length_data, token_data = [], []
	for row in dtitle_reader(FLAGS.input_file, FLAGS.dtitle_schema):
		l = len(row.html)
		length_data.append(l)
	print('Stats of raw html length:')
	print('Start\tFreq')
	hist, bin_edges = np.histogram(length_data, 100)
	for i in range(100):
		print('%d\t%d' % (bin_edges[i], hist[i]))
	print('%d\t-' % bin_edges[100])


def main(_):
	FLAGS = flags.FLAGS
	if FLAGS.cmd == 'pre-process':
		preprocess_raw_input(FLAGS)
	elif FLAGS.cmd == 'build-vocab':
		build_vocab(FLAGS)
	elif FLAGS.cmd == 'tokenize-dtitle':
		tokenize_dtitle(FLAGS)
	elif FLAGS.cmd == 'check-stats':
		check_stats(FLAGS)
	elif FLAGS.cmd == 'print-flags':
		print_flags(FLAGS)


if __name__ == '__main__':
	flags.DEFINE_enum('cmd', None, ['pre-process', 'build-vocab', 'check-stats', 'print-flags', 'tokenize-dtitle'], 'the command to execute')
	flags.mark_flag_as_required('cmd')
	flags.DEFINE_string('input_file', None, 'input dtitle file name for pre-process and build-vocab')
	# params for dtitle_reader
	flags.DEFINE_string('input_schema', 'url,doc_url,visual_title,hostname,title,html', 'input file schema, used fields: url,title,hostname,html')
	flags.DEFINE_string('dtitle_schema', 'url,title,hostname,html', 'dtitle file schema, used by build_vocab() and model training')
	# params for pre-process
	flags.DEFINE_boolean('remove_title', True, 'filter out content in <title> tag')
	flags.DEFINE_boolean('remove_head', True, 'only keep content in <body> tag')
	flags.DEFINE_boolean('check_enoughtokens', False, 'filter out examples whose title doesn''t have enough tokens')
	flags.DEFINE_boolean('check_exactmatch', False, 'filter out examples whose title doesn''t exact-match in html body')
	flags.DEFINE_boolean('check_fuzzymatch', False, 'filter out examples whose title doesn''t fuzzy-match in html body')
	flags.DEFINE_float('max_suppress_ratio', 0.1, 'the max percentage of suppressed examples (title is empty) to generate')
	flags.DEFINE_boolean('suppress_enoughtokens', True, 'filter out examples whose title doesn''t have enough tokens')
	flags.DEFINE_boolean('suppress_exactmatch', True, 'filter out examples whose title doesn''t exact-match in html body')
	flags.DEFINE_boolean('suppress_fuzzymatch', False, 'filter out examples whose title doesn''t fuzzy-match in html body')
	flags.DEFINE_boolean('output_raw_html', False, 'output unmodified html body')
	flags.DEFINE_integer('doc_length_limit', 200*1024, 'max allowed html length')
	# params for build-vocab
	flags.DEFINE_string('vocab_file_prefix', None, 'the prefix of target vocab file for build-vocab')
	flags.DEFINE_integer('target_vocab_size', 8192, 'target vocab size in build-vocab')
	flags.DEFINE_integer('max_subword_length', 16, 'the max token length for building vocab')
	flags.DEFINE_float('max_corpus_chars', 4, 'unit GB(2**30 bytes)')
	# params for tokenize-dtitle
	flags.DEFINE_integer('token_count_limit', 30*1024, 'max allowed token count, 0 means no limit')

	app.run(main)

