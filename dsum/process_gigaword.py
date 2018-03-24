#!/usr/bin/env python3

import sys
import collections

def filter_articles(filefn):
    for l in open(filefn):
        splits = l.strip().split("\t")
        if len(splits) != 3:
            continue
        _, title, article = splits
        title_words = title.split()
        article_words = article.split()

        # No blanks.
        if any((word == "" for word in title_words)):
            continue

        if any((word == "" for word in article_words)):
            continue

        if not any((word == "." for word in article_words)):
            continue

        # Spurious words to blacklist.
        # First set is words that never appear in input and output
        # Second set is punctuation and non-title words.
        bad_words = ['update#', 'update', 'recasts', 'undated', 'grafs', 'corrects',
                    'retransmitting', 'updates', 'dateline', 'writethru',
                    'recaps', 'inserts', 'incorporates', 'adv##',
                    'ld-writethru', 'djlfx', 'edits', 'byline',
                    'repetition', 'background', 'thruout', 'quotes',
                    'attention', 'ny###', 'overline', 'embargoed', ' ap ', ' gmt ',
                    ' adds ', 'embargo',
                    'urgent', '?', ' i ', ' : ', ' - ', ' by ', '-lrb-', '-rrb-']
        if any((bad in title.lower()
                for bad in bad_words)):
            continue

        # Reasonable lengths
        if not (10 < len(article_words) < 1000 and
                3 < len(title_words) < 50):
            continue

        # Some word match.
        matches = len(set([w.lower() for w in title_words if len(w) > 3]) &
                    set([w.lower() for w in article_words if len(w) > 3]))
        if matches < 1:
            continue

        # Okay, print.
        print(l.lower().strip())

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
    for word, count in counter.most_common(200000 - 4):
        print(word, count)
    print('<s> 0')
    print('</s> 0')
    print('<UNK> 0')
    print('<PAD> 0')

if len(sys.argv) != 3:
    exit(-1)

cmd = sys.argv[1]
filefn = sys.argv[2]

if cmd == 'filter':
    filter_articles(filefn)
elif cmd == 'vocab':
    build_vocab(filefn)
else:
    print('Error: invalid command.', cmd)
    exit(-1)
