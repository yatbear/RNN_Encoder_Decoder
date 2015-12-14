#!usr/bin/env python

# Usage: python [files]
#
# RNN Encoder-Decoder for phrase based translation.
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-12-14: Finish the vanilla version.

from rnn_encoder_decoder import *

def main():
    # Read input
    phrase_table = [line for line in open('phrase-table').readlines()]

    x_phrases = [line[0].strip().split() for line in phrase_table]
    y_phrases = [line[1].strip().split() for line in phrase_table]

    # Build source and target vocabularies
    x_vcb = [line.strip().split()[1] for line in open('en.vcb').readlines()]
    y_vcb = [line.strip().split()[1] for line in open('fr.vcb').readlines()]

    # Collect lexical counts
    # x_lexc = [int(line.split()[2]) for line in open('en.vcb').readlines()]
    # y_lexc = [int(line.split()[2]) for line in open('fr.vcb').readlines()]

    # Extract phrase translation probability distributions
    pr_fe = [float(line[2].split()[0]) for line in phrase_table]
    pr_ef = [float(line[2].split()[2]) for line in phrase_table]

    nx, ny = len(x_vcb) + 1, len(y_vcb) + 1 # add OOV to vocab
    
    # Prepare training data
    x_vecs = list() 
    y_vecs = list() 

    # Construct training phrases with one-hot-vectors
    for phrase in x_phrases: 
        vec = np.zeros((len(x_phrases), nx)) 
        for (i, word) in enumerate(phrase):
            j = x_vcb.index(word) if word in x_vcb else nx - 1
            vec[i][j] = 1.0 
        x_vecs.append(vec) 

    # Construct training labels with one-hot-vectors
    for phrase in y_phrases:
        vec = np.zeros((len(y_phrases), ny))
        for (i, word) in enumerate(phrase):
            j = y_vcb.index(word) if word in y_vcb else ny - 1
            vec[i][j] = 1.0
        y_vecs.append(vec) 


    # Prepare data
    train_inds = list()
    test_inds = list()
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

    # Split the data into train/test datasets
    thres = 0.5
    for i, start in enumerate(src_start):
        n = src_num[i]
        if n == 1:
            train_inds.append(start)
        elif max(pr_fe[start:start+n]) < thres or max(pr_ef[start:start+n]) < thres:
            p = (np.array(pr_fe[start:start+n]) + np.array(pr_ef[start:start+n])) / 2.0
            ind = list(p).index(max(p))
            train_inds.append(ind)
            for j in xrange(start, start + n):
                if j != ind:
                    test_inds.append(j)
        else:
            for j in xrange(start, start + n):
                if pr_fe[j] >= thres and pr_ef[j] >= thres:
                    train_inds.append(j)
                else:
                    test_inds.append(j)    

    # Rescore  
    scores = np.zeros(len(x_phrases))
    for i, start in enumerate(src_start):
        # Initialize the model                
        rnn = rnn_encoder_decoder(nx, ny)
        n = src_num[i]
        testlist = list()
        for j in xrange(start, start + n):
            if j in train_inds:
                x, y = x_vecs[j], y_vecs[j]
                scores[j] = rnn.train(x, y)
            else:
                testlist.append(j)
        for j in testlist:
            x, y = x_vecs[j], y_vecs[j]
            scores[j] = rnn.score(x, y)
    
    with open('rescore-table', 'w') as f:
        for i, score in enumerate(scores):
            entry = phrase_table[i][0] + '|||' + phrase_table[i][1] + \
                    '|||' + phrase_table[i][2] + str(score)
            for j in xrange(3, len(phrase_table[i])):
                entry += '|||' + phrase_table[i][j]
            f.write(entry)
    f.close()

if __name__ == '__main__':
    main()