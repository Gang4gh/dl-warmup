#!/usr/bin/env python3

import sys
import re

def convert_rawinput_to_captitles(fn_in):
	for l in open(fn_in, 'r', encoding='utf8'):
		inputs = l.strip().split('\t')
		if len(inputs) != 3:
			print('invalid input, len(inputs)!=3, {}'.format(inputs[0]), file=sys.stderr)
			continue
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

if __name__ == '__main__':
	convert_rawinput_to_captitles(sys.argv[1])

