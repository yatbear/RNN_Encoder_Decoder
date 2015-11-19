#!usr/bin/env python

# Usage: python [files]
#
# RNN Encoder-Decoder for phrase based translation.
#
# Author: <sapphirejyt@gmail.com>
#         2015-11-19: Created for MT independent study.

import theano.tensor as T
import numpy as np

class rnn_encoder_decoder(object):

    def __init__(self, nx, ny, ne=500, nh=1000):
        #  x :: source phrase (nx dimensional one hot vectors)
        #  y :: target phrase (ny dimensional one hot vectors)
        # nx :: source vocabulary size
        # ne :: word embedding dimension
        # nh :: number of hidden units
        # ny :: target vocabulary size

        # Parameters of the RNN encoder
        self.emb = T.shared(name='embeddings',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nx, ne))
                            .astype(T.config.floatX))
        self.Wx = T.shared(name='Wx',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, ne))
                            .astype(T.config.floatX))
        self.Wh_e = T.shared(name='Wh_e',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                            .astype(T.config.floatX))
        self.V_e = T.shared(name='V_e',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                            .astype(T.config.floatX))
        self.h_e_0 = T.shared(name='h_e_0',
                            value=np.zeros(nh,
                            dtype=T.config.floatX))

        # Parameters of the RNN decoder
        self.V_d = T.shared(name='V_d',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                            .astype(T.config.floatX))
        self.Wc = T.shared(name='Wc',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                            .astype(T.config.floatX))
        self.Wh_d = T.shared(name='Wh_d',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh))
                            .astype(T.config.floatX))
        self.Wy = T.shared(name='Wy',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (nh, ne))
                            .astype(T.config.floatX))
        self.y_0 = T.shared(name='y_0',
                            value=np.zeros(ne,
                            dtype=T.config.floatX))

        # Parameters of the output layer
        self.Oh = T.shared(name='Oh',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, nh))
                            .astype(T.config.floatX))
        self.Oy = T.shared(name='Oy',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, ny))
                            .astype(T.config.floatX))
        self.Oc = T.shared(name='Oc',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (2*nh, nh))
                            .astype(T.config.floatX))
        self.Gl = T.shared(name='Gl',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (ny, ne))
                            .astype(T.config.floatX))
        self.Gr = T.shared(name='Gr',
                            value=0.2 * np.random.uniform(-1.0, 1.0, (ne, nh))
                            .astype(T.config.floatX))

        # Bundle
        self.params = [self.emb, self.Wx, self.Wh_e, self.V_e, 
            self.V_d, self.Wh_d, self.Wy, self.Oh, self.Oy, self.Oc, self.Gl, self.Gr]

        # Build input from embedding matrix
        indxs = T.imatrix()
        ex = self.emb[idxs].reshape((idxs.shape[0], ne))


    # Encode an input phrase into a summary vector
    def encode(x):
        # Construct recursion
        def en_recurrence(x_t, h_e_tm1):
            h_e_t = T.tanh(T.dot(self.Wx, x_t) 
                        + T.dot(self.Wh_e, h_e_tm1))
            return h_e_t

        # Compute the encoder hidden state recursively
        h_e, _ = T.scan(fn=en_recurrence, 
                        sequence=x, 
                        outputs_info=self.h_e_0,
                        n_steps=x.shape[0])

        # Compute the summary vector
        c = T.tanh(T.dot(self.V_e, h_e[-1]))

        return c


    def decode(x, y):
        # Get the summary vector
        c = encode(x)
        # Initialize the decoder hidden state
        h_d_0 = T.tanh(T.dot(self.V_d, c))

        # Construct recursion
        def de_recurrence(h_d_tm1, c, y_t):
            # Compute hidden layer
            h_t = T.tanh(T.dot(self.Wc, c) 
                        + T.dot(self.Wh_d, h_d_tm1)
                        + T.dot(self.Wy, y_t))
            
            # Compute output layer
            ss_t = T.dot(self.Oh, h_t) + T.dot(self.Oy, y_tm1), T.dot(self.Oc, c)

            # Compute maxout units
            for i in xrange(nh):
                s_t[i] = T.max(ss_t[2*i:2*(i+1)])

            # Compute probability of generating the target phrase
            G = T.dot(self.Gl, self.Gr)
            p_t = T.nnet.softmax(T.dot(G, s_t))

            # Compute the negative log-likelihood
            nll_t = -T.log(p_t)

            return [h_t, nll_t]

        [h, nll], _ = T.scan(fn=de_recurrence,
                            outputs_info=[h_d_0, None],
                            non_sequences=[c, self.y_0],
                            n_steps=y.shape[0])
