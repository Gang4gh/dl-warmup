#!/usr/bin/env python3

import sys
import re
import time


def dtitle_reader(dtitle_file, log_per_n_step=None):
	lcount = 0
	for l in open(dtitle_file, 'r', encoding='utf8'):
		inputs = l.strip().split('\t')
		if len(inputs) != 3:
			print('invalid input, len(inputs)!=3, {}'.format(inputs[0]), file=sys.stderr)
			continue
		yield inputs
		if log_per_n_step:
			lcount += 1
			if lcount % log_per_n_step == 0:
				print('read {}k examples from {}'.format(lcount//1024, dtitle_file), file=sys.stderr)


def preprocess_raw_input(input_file):
	for inputs in dtitle_reader(input_file):
		inputs[1] = re.sub(r'#N#|#R#|#TAB#', ' ', inputs[1])
		inputs[1] = re.sub(r' +', ' ', inputs[1])
		inputs[2] = re.sub(r'</html>.*', '</html>', inputs[2], flags=re.I)
		inputs[2] = re.sub(r'#N#|#R#|#TAB#', ' ', inputs[2])
		inputs[2] = re.sub(r' +', ' ', inputs[2])
		m = re.match(r'.{,2048}[^\w&</]', inputs[2])
		if m:
			inputs[2] = m.group(0).strip()
			print('\t'.join(inputs))
		else:
			print('invalid input, no-match, {}'.format(inputs[0]), file=sys.stderr)
			continue


def build_vocab(input_file, vocab_file_prefix, target_vocab_size):
	import tensorflow_datasets as tfds
	target_vocab_file = '{}-{}'.format(vocab_file_prefix, target_vocab_size)
	print('{}: start to build a subwords tokenizer({}).'.format(time.asctime(), target_vocab_file))
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
		(s.encode() for inputs in dtitle_reader(input_file, 100*1024) for s in inputs[1:]),
		target_vocab_size=int(target_vocab_size),
		max_subword_length=16)
	tokenizer.save_to_file(target_vocab_file)
	print('{}: the subwords tokenizer({}) is ready.'.format(time.asctime(), target_vocab_file))


if __name__ == '__main__':
	cmd = sys.argv[1] if len(sys.argv) > 1 else None
	if cmd == 'pre-process':
		preprocess_raw_input(*sys.argv[2:])
	elif cmd == 'build-vocab':
		build_vocab(*sys.argv[2:])
	elif cmd == 'build-tfexample':
		pass

