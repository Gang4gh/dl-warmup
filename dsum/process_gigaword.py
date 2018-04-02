#!/usr/bin/env python3

import sys
import re
import collections

def _remove_digits(text):
    return re.sub(r'\d[,\\/\d]*', '#', text)

def filter_articles(filefn, argv):
    keep_sentence_count = 100 if len(argv) < 1 else int(argv[0])
    max_word_len = 1000 if len(argv) < 2 else int(argv[1])
    max_title_word_len = 50 if len(argv) < 3 else int(argv[2])

    for l in open(filefn):
        splits = _remove_digits(l.strip()).lower().split("\t")
        if len(splits) != 3:
            continue
        doc_id, title, article = splits
        title_words = title.split()
        article_words = ' <eos/> '.join(article.split(' <eos/> ')[:keep_sentence_count]).split()

        # reasonable lengths
        if not (10 < len(article_words) <= max_word_len and
                3 < len(title_words) < max_title_word_len):
            continue

        print('\t'.join([doc_id, title, ' '.join(article_words)]))

def build_vocab(filefn):
    counter = collections.Counter()

    for l in open(filefn):
        splits = l.strip().split("\t")
        if len(splits) != 3:
            continue
        _, title, article = splits
        counter.update(title.split())
        counter.update(article.split())
    
    del counter['<eos/>']

    print('<UNK> 0')
    print('<PAD> 0')
    print('<s> 0')
    print('</s> 0')
    for word, count in counter.most_common(100000 - 4):
        print(word, count)

cmd = sys.argv[1]
filefn = sys.argv[2]

if cmd == 'filter':
    filter_articles(filefn, sys.argv[3:])
elif cmd == 'vocab':
    build_vocab(filefn)
else:
    print('Error: invalid command.', cmd)
    exit(-1)
