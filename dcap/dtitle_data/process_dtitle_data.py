#!/usr/bin/env python3

import sys
import re
import time


def dtitle_reader(dtitle_file):
	for l in open(dtitle_file, 'r', encoding='utf8'):
		inputs = l.strip().split('\t')
		if len(inputs) != 3:
			print('invalid input, len(inputs)!=3, {}'.format(inputs[0]), file=sys.stderr)
			continue
		yield inputs


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


def build_vocab(input_file, vocab_file):
	import tensorflow_datasets as tfds
	print('{}: start of building a subwords tokenizer.'.format(time.asctime()))
	tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
		(s.encode() for inputs in dtitle_reader(input_file) for s in inputs[1:]),
		target_vocab_size=2**13)
	tokenizer.save_to_file(vocab_file)
	print('{}: prepare tokenizers and save to "{}".'.format(time.asctime(), vocab_file))

if __name__ == '__main__':
	cmd = sys.argv[1] if len(sys.argv) > 1 else None
	if cmd == 'pre-process':
		preprocess_raw_input(*sys.argv[2:])
	elif cmd == 'build-vocab':
		build_vocab(*sys.argv[2:])
	elif cmd == 'build-tfexample':
		pass

