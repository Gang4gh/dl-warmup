#!/usr/bin/env python3

import sys
import re
import time
import gzip
import collections
from multiprocessing import Pool

from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds

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
	dtitle_schema_columns = FLAGS.dtitle_schema.split(',')

	total, valid, suppressed = 0, 0, 0
	for row in dtitle_reader(FLAGS.input_file, FLAGS.input_schema):
		total += 1
		url, title, html = row.Url, row.AHtmlTitle, row.CleanedHtmlBody
		if not url or not html: continue

		#html = re.sub(r'</html>.*', '</html>', html, flags=re.I)

		# apply html modification (mask) options to modify content
		if FLAGS.mask_html_title:
			html = re.sub(r'<title.*?</title>', ' ', html, flags=re.I)
		if FLAGS.mask_title_fields:
			html = re.sub(r'<meta[^>]*=["\'](?:og:|og&#x3a;)?title["\'][^>]*>', '', html, flags=re.I)
		if FLAGS.mask_description_fields:
			html = re.sub(r'<meta[^>]*=["\'](?:og:|og&#x3a;)?description["\'][^>]*>', '', html, flags=re.I)
		if FLAGS.mask_og_sitename:
			html = re.sub(r'<meta[^>]*=["\'](?:og:|og&#x3a;)?site_name["\'][^>]*>', '', html, flags=re.I)
		if False:
			m = re.search(r'<meta[^>]*=["\']description["\'][^>]*>', html, flags=re.I)
			mstring = m.group(0) if m else None
			print(f'url = {url}\nhtml = {html[:1000]}\ntmstring = {mstring}')
			if mstring: input("Press Enter to continue...")
			continue

		# split and truncate head and body
		head_regex = r'<head\W.*?</head>'
		htmlhead = ' '.join(re.findall(head_regex, html, flags=re.I))
		htmlbody = re.sub(head_regex, '', html, flags=re.I)

		htmlhead = htmlhead[:FLAGS.htmlhead_length_limit]
		htmlbody = htmlbody[:FLAGS.htmlbody_length_limit]

		# apply filtering options
		title_normalized, htmlbody_normalized = (_normalize_string(s) for s in [title, htmlbody])
		title_tokens = [w for w in re.split(r'\W+', title_normalized) if w]
		if FLAGS.filter_short_title and len(title_tokens) <= 1:
			continue

		if not FLAGS.for_inference and (FLAGS.suppress_enoughtokens and len(title_tokens) <= 1
			or FLAGS.suppress_exactmatch and title_normalized not in htmlbody_normalized
			or FLAGS.suppress_fuzzymatch and any(token not in htmlbody_normalized for token in title_tokens)):
			if  FLAGS.max_suppress_ratio * (valid + suppressed) >= suppressed:
				suppressed += 1
				title = ''
			else:
				continue

		# output by the order defined in dtitle_schema
		res = []
		for col in dtitle_schema_columns:
			if col == 'TargetTitle':
				res.append(title)
			elif col == 'HtmlBody':
				res.append(htmlbody)
			elif col == 'HtmlHead':
				res.append(htmlhead)
			else:
				res.append(getattr(row, col))
		print('\t'.join([_normalize_string(s, toLower=False) for s in res]))
		valid += 1 if title else 0

	ignored = total - valid - suppressed
	print(f'processed {total} example(s), including {valid} ({valid/total*100:.2f}%) valid, {suppressed} ({suppressed/total*100:.2f}%) suppressed and {ignored} ({ignored/total*100:.2f}) ignored examples, from {FLAGS.input_file}', file=sys.stderr)


def build_vocab(FLAGS):
	def _get_vocab_corpus():
		from pathlib import Path
		columns = [col.split(':') for col in FLAGS.vocab_corpus_columns.split(',')]
		column_with_limits = [(col[0], int(col[1]) if len(col) > 1 else 128) for col in columns]

		quota = int(FLAGS.max_corpus_chars*(2**30))
		for fp in sorted(Path('.').glob(FLAGS.input_file)):
			print(f'read from {fp} with quota={quota//(1024*1024)}MB, at {time.asctime()}')
			for row in dtitle_reader(str(fp), FLAGS.input_schema, 100*1024):
				for col in column_with_limits:
					text = getattr(row, col[0])[:col[1]]
					if text:
						#print(f'col={col}, len={len(text)}, text={text[:100]}')
						btext = _normalize_string(text).encode()
						yield btext
						quota -= len(btext)
						if quota < 0:
							print(f'reach limit and stop.')
							return

	target_vocab_file = '{}-{}'.format(FLAGS.vocab_file_prefix, FLAGS.target_vocab_size)
	print('{}: start to build a subwords tokenizer({}) with max_subword_length={}, max_corpus_chars={}GB.'.format(time.asctime(), target_vocab_file, FLAGS.max_subword_length, FLAGS.max_corpus_chars))

	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(_get_vocab_corpus(),
			target_vocab_size = FLAGS.target_vocab_size,
			max_subword_length = FLAGS.max_subword_length,
			max_corpus_chars = int(FLAGS.max_corpus_chars*(2**30)),
			reserved_tokens = ['<EOS>'] + [f'<BOS#{i}>' for i in range(10)] + [f'<EOS#{i}>' for i in range(10)])
	tokenizer.save_to_file(target_vocab_file)
	_save_FLAGS_and_code(FLAGS, target_vocab_file + '.log')
	print('{}: the subwords tokenizer({}) is ready.'.format(time.asctime(), target_vocab_file))


def tokenize_dtitle(FLAGS):
	vocab_file = '{}-{}'.format(FLAGS.vocab_file_prefix, FLAGS.target_vocab_size)
	tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
	print(f'initilize tokenizer by vocab file [{vocab_file}].')

	assert FLAGS.input_file.endswith('.dtitle.gz')
	tfrecord_file = FLAGS.input_file[:-10] + '.tokenized-tfrecord'
	if FLAGS.compression_type == 'GZIP':
		tfrecord_file += '.gz'

	stats = []
	def _create_int64List_feature(text, limit):
		arr = tokenizer.encode(text)
		stats.append(len(arr))
		if limit:
			arr = arr[:limit]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))

	with tf.io.TFRecordWriter(tfrecord_file, FLAGS.compression_type) as tfwriter:
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


tokenizer = None
def _create_example(row):
	def _create_int64List_feature(text, limit):
		arr = tokenizer.encode(_normalize_string(text))
		if limit: arr = arr[:limit]
		return tf.train.Feature(int64_list=tf.train.Int64List(value=arr))

	url, title, hostname, html = row
	example = {
		'url': _create_int64List_feature(url, 0),
		'title': _create_int64List_feature(title, 0),
		'hostname': _create_int64List_feature(hostname, 0),
		'html': _create_int64List_feature(html, flags.FLAGS.token_count_limit),
	}
	return tf.train.Example(features=tf.train.Features(feature=example)).SerializeToString()


def tokenize_dtitle_mp(FLAGS):
	global tokenizer
	vocab_file = '{}-{}'.format(FLAGS.vocab_file_prefix, FLAGS.target_vocab_size)
	tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
	print(f'initilize tokenizer by vocab file [{vocab_file}].')

	assert FLAGS.input_file.endswith('.dtitle.gz')
	tfrecord_file = FLAGS.input_file[:-10] + '.tokenized-tfrecord'
	if FLAGS.compression_type == 'GZIP':
		tfrecord_file += '.gz'

	count = 0
	with tf.io.TFRecordWriter(tfrecord_file, FLAGS.compression_type) as tfwriter, Pool() as pool:
		for proto in pool.imap(_create_example, ((row.Url, row.TargetTitle, row.InjHdr_CDG_H, row.HtmlBody) for row in dtitle_reader(FLAGS.input_file, FLAGS.dtitle_schema))):
			count += 1
			tfwriter.write(proto)
	print(f'complete tokenization of {FLAGS.input_file}, token limit = {FLAGS.token_count_limit}. write {count} records to {tfrecord_file}.')


def print_flags(FLAGS, file=None):
	print('FLAGS:', file=file)
	for f in FLAGS.get_key_flags_for_module(__file__):
		print('    {}: {}'.format(f.name, f.value), file=file)


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


def _normalize_string(s, toLower=True):
	s = re.sub(r'\s+', ' ', s).strip()
	return s.lower() if toLower else s


def _save_FLAGS_and_code(FLAGS, filename):
	with open(filename, 'w') as logfile:
		logfile.write('-- FLAGS --\n')
		print_flags(FLAGS, logfile)
		logfile.write('-- source code --\n')
		with open(__file__) as fi:
			for line in fi: logfile.write(line)


def main(_):
	FLAGS = flags.FLAGS
	if FLAGS.cmd == 'pre-process':
		preprocess_raw_input(FLAGS)
	elif FLAGS.cmd == 'build-vocab':
		build_vocab(FLAGS)
	elif FLAGS.cmd == 'tokenize-dtitle':
		tokenize_dtitle(FLAGS)
	elif FLAGS.cmd == 'tokenize-dtitle-mp':
		tokenize_dtitle_mp(FLAGS)
	elif FLAGS.cmd == 'check-stats':
		check_stats(FLAGS)
	elif FLAGS.cmd == 'print-flags':
		print_flags(FLAGS)


if __name__ == '__main__':
	flags.DEFINE_enum('cmd', None, ['pre-process', 'build-vocab', 'check-stats', 'print-flags', 'tokenize-dtitle', 'tokenize-dtitle-mp'], 'the command to execute')
	flags.mark_flag_as_required('cmd')
	flags.DEFINE_string('input_file', None, 'input dtitle file name for pre-process and build-vocab')
	# params for dtitle_reader
	flags.DEFINE_string('input_schema', 'Url,DocumentUrl,HostName,IsSiteHomepage,VisualTitle,InjHdr_CDG_1,InjHdr_CDG_2,InjHdr_CDG_H,InjHdr_CDG_E,BrokenUrl1,BrokenUrl2,BrokenUrl3,AnnotationDesc,AHtmlTitle,AOGTitle,AOGDesc,AOGSiteName,AMetaDesc,Editorial_Name,Wiki_Name,Entity_Name,ODPTitle,ODPDescription,CleanedHtmlBody,RandomValue', 'input file schema, used fields: url,title,hostname,html')
	flags.DEFINE_string('dtitle_schema', 'Url,DocumentUrl,HostName,IsSiteHomepage,VisualTitle,InjHdr_CDG_1,InjHdr_CDG_2,InjHdr_CDG_H,InjHdr_CDG_E,BrokenUrl1,BrokenUrl2,BrokenUrl3,AHtmlTitle,AOGTitle,AOGDesc,AOGSiteName,AMetaDesc,Editorial_Name,Wiki_Name,Entity_Name,ODPTitle,ODPDescription,TargetTitle,HtmlHead,HtmlBody', 'input file schema, used fields: url,title,hostname,html')
	# params for pre-process
	flags.DEFINE_boolean('mask_html_title', True, 'remove content in <title> tag (html_title) from html')
	flags.DEFINE_boolean('mask_title_fields', False, 'remove meta-title, og-title from html')
	flags.DEFINE_boolean('mask_description_fields', False, 'remove meta-description, og-description from html')
	flags.DEFINE_boolean('mask_og_sitename', False, 'remove og_sitename from html')
	flags.DEFINE_boolean('filter_short_title', False, 'filter out examples whose title doesn''t have enough tokens')
	flags.DEFINE_float('max_suppress_ratio', 0.1, 'the max percentage of suppressed examples (title is empty) to generate')
	flags.DEFINE_boolean('suppress_enoughtokens', True, 'filter out examples whose title doesn''t have enough tokens')
	flags.DEFINE_boolean('suppress_exactmatch', True, 'filter out examples whose title doesn''t exact-match in html body')
	flags.DEFINE_boolean('suppress_fuzzymatch', False, 'filter out examples whose title doesn''t fuzzy-match in html body')
	flags.DEFINE_integer('htmlhead_length_limit', 10*1024, 'max allowed html head length')
	flags.DEFINE_integer('htmlbody_length_limit', 100*1024, 'max allowed html body length')
	flags.DEFINE_integer('default_length_limit', 1*1024, 'max allowed length other than htmlhead/body')
	flags.DEFINE_boolean('for_inference', False, 'when its'' True, by pass some filtering logic in data pre-process')
	# params for build-vocab
	flags.DEFINE_string('vocab_corpus_columns', 'Url:256,InjHdr_CDG_H,InjHdr_CDG_E,BrokenUrl1:256,AHtmlTitle,AOGSiteName,AMetaDesc:512,Editorial_Name,Wiki_Name,Entity_Name,CleanedHtmlBody:40960',
			'list of column_name:length_limit to build vocab, default length_limit is 128')
	flags.DEFINE_string('vocab_file_prefix', None, 'the prefix of target vocab file for build-vocab')
	flags.DEFINE_integer('target_vocab_size', 8192, 'target vocab size in build-vocab')
	flags.DEFINE_integer('max_subword_length', 16, 'the max token length for building vocab')
	flags.DEFINE_float('max_corpus_chars', 4, 'unit GB(2**30 bytes)')
	# params for tokenize-dtitle
	flags.DEFINE_integer('token_count_limit', 30*1024, 'max allowed token count, 0 means no limit')
	flags.DEFINE_enum('compression_type', 'GZIP', ['', 'GZIP'], 'compression type used for tfrecord files')

	app.run(main)

