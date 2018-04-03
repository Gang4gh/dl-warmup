#!/usr/bin/env python3

import sys
import re
import collections

import data


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
        article_words = data.EOS_TOKEN2.join(article.split(data.EOS_TOKEN2)[:keep_sentence_count]).split()

        # reasonable lengths
        if not (10 < len(article_words) <= max_word_len and
                3 < len(title_words) < max_title_word_len):
            continue

        print('\t'.join([doc_id, title, ' '.join(article_words)]))


def build_vocab(filefn, max_word_count):
    counter = collections.Counter()

    for l in open(filefn):
        splits = l.strip().split("\t")
        if len(splits) != 3:
            continue
        _, title, article = splits
        counter.update(title.split())
        counter.update(article.split())
    
    del counter[data.EOS_TOKEN]

    print(data.UNKNOWN_TOKEN, 0)
    print(data.PAD_TOKEN, 0)
    print(data.SENTENCE_START, 0)
    print(data.SENTENCE_END, 0)
    for word, count in counter.most_common(max_word_count - 4):
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


if __name__ == '__main__':
	#main():
    if len(sys.argv) < 3:
        print('Error: invalid argv.', sys.argv)
        exit(-1)

    cmd = sys.argv[1]
    filefn = sys.argv[2]

    if cmd == 'filter':
        filter_articles(filefn, sys.argv[3:])
    elif cmd == 'vocab':
        build_vocab(filefn, 10000)
    elif cmd == 'count-title':
        count_titles(filefn, 1)
    else:
        print('Error: invalid command.', cmd)
        exit(-1)
