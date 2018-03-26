#!/usr/bin/env python3

import sys
import collections

def filter_articles(filefn, argv):
    keep_sentence_count = 100 if len(argv) < 1 else int(argv[0])
    max_word_len = 1000 if len(argv) < 2 else int(argv[1])
    max_title_word_len = 50 if len(argv) < 3 else int(argv[2])

    for l in open(filefn):
        splits = l.strip().split("\t")
        if len(splits) != 3:
            continue
        doc_id, title, article = splits
        title_words = title.split()
        article_words = ' <eos/> '.join(article.split(' <eos/> ')[:keep_sentence_count]).split()

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
        if not (10 < len(article_words) <= max_word_len and
                3 < len(title_words) < max_title_word_len):
            continue

        # Some word match.
        matches = len(set([w.lower() for w in title_words if len(w) > 3]) &
                    set([w.lower() for w in article_words if len(w) > 3]))
        if matches < 1:
            continue

        # Okay, print.
        #print(l.lower().strip())
        print('\t'.join([doc_id, title, ' '.join(article_words)]).lower())

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
    for word, count in counter.most_common(100000 - 4):
        print(word, count)
    print('<UNK> 0')
    print('<s> 0')
    print('</s> 0')
    print('<PAD> 0')

cmd = sys.argv[1]
filefn = sys.argv[2]

if cmd == 'filter':
    filter_articles(filefn, sys.argv[3:])
elif cmd == 'vocab':
    build_vocab(filefn)
else:
    print('Error: invalid command.', cmd)
    exit(-1)
