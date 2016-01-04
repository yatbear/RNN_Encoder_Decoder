#!usr/bin/env python

# Usage: python [files]
#
# RNN Encoder-Decoder for phrase based translation.
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-11-19: Created for MT independent study.
#         2015-12-10: Fixed major bugs.
#         2015-12-14: Finished the vanilla version.

import theano
import theano.tensor as T
import theano.tensor.signal.downsample as ds
import numpy as np
from collections import OrderedDict

class rnn_encoder_decoder(object):

    def __init__(self, nx, ny, ne=100, nh=1000, lr=0.01):
        # nx :: source vocabulary size
        # ne :: word embedding dimension
        # nh :: number of hidden units
        # ny :: target vocabulary size
        # lr :: learning rate

        # Parameters of the RNN encoder
        self.emb = theano.shared(name='embeddings',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nx, ne))
                                .astype(theano.config.floatX))
        self.Wx = theano.shared(name='Wx',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, ne))
                                .astype(theano.config.floatX))
        self.Wh_e = theano.shared(name='Wh_e',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.V_e = theano.shared(name='V_e',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.bh_e = theano.shared(name='bh_e',
                                value=np.zeros(nh,
                                dtype=theano.config.floatX))
        self.h_e_0 = theano.shared(name='h_e_0',
                                value=np.zeros(nh, 
                                dtype=theano.config.floatX))

        # Parameters of the RNN decoder
        self.V_d = theano.shared(name='V_d',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.Wc = theano.shared(name='Wc',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.Wh_d = theano.shared(name='Wh_d',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.Wy = theano.shared(name='Wy',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, ny))
                                .astype(theano.config.floatX))
        self.bh_d = theano.shared(name='bh_d',
                                value=np.zeros(nh,
                                dtype=theano.config.floatX))
        self.y_0 = theano.shared(name='y_0',
                                value=np.zeros(ny,
                                dtype=theano.config.floatX))

        # Parameters of the output layer
        self.Oh = theano.shared(name='Oh',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        self.Oy = theano.shared(name='Oy',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, ny))
                                .astype(theano.config.floatX))
        self.Oc = theano.shared(name='Oc',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                                .astype(theano.config.floatX))
        ## For max pooling computation
        # self.Oh = theano.shared(name='Oh',
        #                         value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, nh))
        #                         .astype(theano.config.floatX))
        # self.Oy = theano.shared(name='Oy',
        #                         value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, ny))
        #                         .astype(theano.config.floatX))
        # self.Oc = theano.shared(name='Oc',
        #                         value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, nh))
        #                         .astype(theano.config.floatX))
        self.Gl = theano.shared(name='Gl',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (ny, ne))
                                .astype(theano.config.floatX))
        self.Gr = theano.shared(name='Gr',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (ne, nh))
                                .astype(theano.config.floatX))

        # Bundle
        self.params = [self.emb, self.Wx, self.Wh_e, self.V_e, self.bh_e, self.V_d, 
                    self.Wh_d, self.Wy, self.bh_d, self.Oh, self.Oy, self.Oc, self.Gl, self.Gr]
        self.names = ['embeddings', 'Wx', 'Wh_e', 'V_e', 'bh_e', 'V_d', 
                    'Wh_d', 'Wy', 'bh_d', 'Oh', 'Oy', 'Oc', 'Gl', 'Gr']
        
        x_seq = T.fmatrix('x_seq')
        y_seq = T.fmatrix('y_seq')

        # Encode an input phrase into a summary vector
        def encode(x_seq):   
            # Build input from embedding matrix
            x = T.dot(x_seq, self.emb)

            # Construct encoder recursion 
            def en_recurrence(x_t, h_e_tm1):
                h_e_t = T.tanh(T.dot(self.Wx, x_t) 
                            + T.dot(self.Wh_e, h_e_tm1) + self.bh_e)
                return h_e_t

            # Compute the encoder hidden state recursively
            h_e, _ = theano.scan(fn=en_recurrence, 
                                sequences=x, 
                                outputs_info=self.h_e_0,
                                n_steps=x.shape[0])

            # Compute the summary vector
            c = T.tanh(T.dot(self.V_e, h_e[-1]))
            return c

        # Decode the summary vector into a target sequence
        def decode(c, y_seq):
            # Initialize the decoder hidden state
            self.h_d_0 = T.tanh(T.dot(self.V_d, c))

            # Construct decoder recursion
            def de_recurrence(t, c, y_seq):
                y_t = y_seq[t]
                y_tm1 = y_seq[t-1] if t == 0 else self.y_0

                # Compute hidden layer
                h_d_t = T.tanh(T.dot(self.Wc, c) 
                            + T.dot(self.Wh_d, self.h_d_0)
                            + T.dot(self.Wy, y_t) + self.bh_d)
                self.h_d_0 = h_d_t[-1]
            
                # Compute output layer
                ss_t = T.dot(self.Oh, h_d_t) + T.dot(self.Oy, y_tm1) + T.dot(self.Oc, c) 

                # Compute maxout units
                # s_t = ds.max_pool_2d(ss_t, (1, 2), ignore_border=True) 

                # Compute probability of generating the target phrase
                G = T.dot(self.Gl, self.Gr)
                p_t = T.nnet.softmax(T.flatten(T.dot(G, ss_t)))

                # Compute the negative log-likelihood
                nll_t = -T.log(p_t)
                return nll_t

            nll, _ = theano.scan(fn=de_recurrence,
                                    sequences=T.arange(y_seq.shape[0]),
                                    outputs_info=None,
                                    non_sequences=[c, y_seq],
                                    n_steps=y_seq.shape[0])
            return T.mean(nll)

        # Get the summary vector
        c = encode(x_seq)

        # Get the negative log-likelihood
        seq_nll = decode(c, y_seq) 
        # print seq_nll.eval()

        # Compute all the gradients automatically to maximize the log-likelihood     
        seq_gradients = T.grad(seq_nll, self.params)
        seq_updates = OrderedDict((p, p - lr*g)
                                    for p, g in zip(self.params, seq_gradients))

        self.train_pair = theano.function(inputs=[x_seq, y_seq],
                                        outputs=seq_nll,
                                        on_unused_input='ignore',
                                        allow_input_downcast=True,
                                        updates=seq_updates)
        self.score_pair = theano.function(inputs=[x_seq, y_seq], 
                                        on_unused_input='ignore', 
                                        allow_input_downcast=True, 
                                        outputs=seq_nll)
     
    def train(self, x_seq, y_seq):
        nll = self.train_pair(x_seq, y_seq)
        return nll

    def score(self, x_seq, y_seq):
        nll = self.score_pair(x_seq, y_seq)
        return nll

    def save(self, path):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(path, name + '.npy'), param.get_value())
            