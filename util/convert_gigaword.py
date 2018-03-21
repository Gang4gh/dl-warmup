#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

#/usr/bin/env python

import sys
import os
import re
import gzip
#@lint-avoid-python-3-compatibility-imports

# Make directory for output if it doesn't exist

try:
    os.mkdir(sys.argv[2] + "/" + sys.argv[1].split("/")[-2])
except OSError:
    pass

# Strip off .gz ending
end = "/".join(sys.argv[1].split("/")[-2:])[:-len(".xml.gz")] + ".txt"

out = open(sys.argv[2] + end, "w")

filename = sys.argv[1].split("/")[-1]

# Parse and print titles and articles
NONE, HEAD, NEXT, TEXT = 0, 1, 2, 3
MODE = NONE
title_parse = ""
article_parse = []

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

lc = 0
doc_id = ''
for l in gzip.open(sys.argv[1]):
    lc += 1
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
        if "(. .)" not in article_parse[0]:
            print lc, article_parse[0]

        article = "(TOP " + " (EOS <eos/>) ".join(article_parse) + ")"
        # title_parse \t article_parse \t title \t article
        print >>out, "\t".join([#title_parse, article,
                                doc_id, filename,
                                " ".join(get_words(title_parse)),
                                " ".join(get_words(article))])
        article_parse = []
        MODE = NONE
    elif MODE == TEXT:
        if ls == "<P>" or ls == "</P>": continue
        article_parse.append(remove_digits(fix_paren(ls)))
