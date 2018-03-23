#!/usr/bin/env python

import sys
import os
import re
import gzip

inputfn = sys.argv[1]
outputfn = sys.argv[2]
outdir = os.path.dirname(outputfn)

# Make directory for output if it doesn't exist
if not os.path.isdir(outdir):
    try:
        os.makedirs(outdir)
    except OSError:
        pass

out = open(outputfn, "w")

# Parse and print titles and articles
NONE, HEAD, NEXT, TEXT = 0, 1, 2, 3
MODE = NONE
title_parse = ''
article_parse = []
doc_id = ''

# FIX: Some parses are mis-parenthesized.
def fix_paren(parse):
    if len(parse) < 2:
        return parse
    if parse[0] == "(" and parse[1] == " ":
        return parse[2:-1]
    return parse

def get_words(parse):
    words = []
    for w in parse.split():
        if w[-1] == ')':
            words.append(w.strip(")"))
            #if words[-1] == ".": break
    return words

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)

line_count = 0
for l in gzip.open(inputfn):
    line_count += 1
    #if line_count > 10000: break
    ls = l.strip()

    if l.find('<DOC id="') == 0:
        doc_id = l.split('"')[1]
    elif MODE == NONE and l.strip() == "<HEADLINE>":
        MODE = HEAD
    elif MODE == HEAD:
        title_parse = remove_digits(fix_paren(l.strip()))
        MODE = NEXT
    elif MODE == NEXT and ls == "<TEXT>":
        MODE = TEXT
    elif MODE == TEXT and ls == "</TEXT>":
        #if "(. .)" not in article_parse[0]:
        #    print line_count, article_parse[0]

        article = "(TOP " + " (EOS <eos/>) ".join(article_parse) + ")"
        # schema: doc_it \t title \t article
        print >>out, "\t".join([doc_id,
                                " ".join(get_words(title_parse)),
                                " ".join(get_words(article))])
        title_parse = ''
        article_parse = []
        doc_id = ''
        MODE = NONE
    elif MODE == TEXT:
        if ls == "<P>" or ls == "</P>": continue
        article_parse.append(remove_digits(fix_paren(ls)))
