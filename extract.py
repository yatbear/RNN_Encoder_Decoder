#!usr/bin/env python

# Usage: python [files]
#
# Data Preparation
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-12-21

import scipy.io as sio

phrase_table = [line for line in open('phrase-table').readlines()]
phrase_pairs = [line.split('|||') for line in phrase_table]
x_phrases = [line[0].strip().split() for line in phrase_pairs]

# Extract phrase translation probability distributions
pr_fe = [float(line[2].split()[0]) for line in phrase_pairs]
pr_ef = [float(line[2].split()[2]) for line in phrase_pairs]

src_start = list()
src_num = list()
src = None
for i, x in enumerate(x_phrases):
    if src != x:
        src = x
        src_start.append(i)
        if i > 0:
            if len(src_num) == 0:
                src_num.append(i)
            else:
                src_num.append(i - src_start[-2])
src_num.append(len(x_phrases) - src_start[-1])

# Truncate the training phrases
small_phrase_table = list()
thres = 0.1
for i, start in enumerate(src_start):
    n = src_num[i]
    if max(pr_fe[start:start+n]) < thres or max(pr_ef[start:start+n]) < thres:
        for j in xrange(start, start+n):
            small_phrase_table.append(phrase_table[j])

with open('small_phrase_table', 'w') as f:
    for line in small_phrase_table:
        f.write(line)
