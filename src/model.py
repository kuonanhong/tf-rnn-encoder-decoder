from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

k = 15000

graph = tf.Graph()
with graph.as_default():
    # Variables
    # Encoder input
    N = tf.placeholder(tf.float32, shape=(), name="N")
    X = tf.placeholder(tf.float32, shape=[N, k])
    we = tf.Variable(tf.truncated_normal([500, k], -0.1, 0.1))
    # Encoder update gate
    wz = tf.Variable(tf.truncated_normal([1000, 500], -0.1, 0.1))
    uz = tf.Variable(tf.truncated_normal([1000, 1000], -0.1, 0.1))
    # Encoder reset gate
    wr = tf.Variable(tf.truncated_normal([1000, 500], -0.1, 0.1))
    ur = tf.Variable(tf.truncated_normal([1000, 1000], -0.1, 0.1))
    # Encoder h~ [find name]
    w = tf.Variable(tf.truncated_normal([1000, 500], -0.1, 0.1))
    u = tf.Variable(tf.truncated_normal([1000, 1000], -0.1, 0.1))
    # Encoder representation weight
    v = tf.Variable(tf.truncated_normal([1000, 1000], -0.1, 0.1))
    # Decoder input
    w_prime_e = tf.Variable(tf.truncated_normal([500, k], -0.1, 0.1))
    # Decoder representation weight
    v_prime = tf.Variable(tf.truncated_normal([1000, 1000], -0.1, 0.1))

    # Encoder
    ht_previous = tf.zeros([1000])
    for t in range(N):
        # Current vector and its embedding
        xt = tf.reshape(tf.slice(N, [t, 0], [1, k]), [k])
        e = tf.matmul(we, xt)
        # Vectors for reset calculation
        wr_e = tf.matmul(wr, e)
        ur_ht_previous = tf.matmul(ur, ht_previous)
        # Vectors for update calculation
        wz_e = tf.matmul(wz, e)
        uz_ht_previous = tf.matmul(wz, ht_previous)

        # Reset calculation
        r = tf.zeros([1000])
        for j in range(1000):
            rj = tf.sigmoid(tf.slice(wr_e, [j], [1]) + tf.slice(ur_ht_previous, [j], [1]))
            r = r + tf.sparse_tensor_to_dense(tf.SparseTensor([j], [rj], [1000]))

        # Vectors for h~ calculation
        w_e = tf.matmul(w, e)
        r_ewm_h_previous = tf.zeros([1000])
        for j in range (1000):
            ewm = tf.slice(r, [j], [1]) * tf.slice(ht_previous, [j], [1])
            r_ewm_h_previous = r_ewm_h_previous + tf.sparse_tensor_to_dense(tf.SparseTensor([j], [ewm], [1000]))
        u_r_ewm_h_previous = tf.matmul(u, r_ewm_h_previous)

        # Hidden calculation
        ht = tf.zeros([1000])
        for j in range(1000):
            #Update calculation
            zj = tf.sigmoid(tf.slice(wz_e, [j], [1]) + tf.slice(uz_ht_previous, [j], [1]))
            #h~ calculation
            hj_tilde = tf.tanh(tf.slice(w_e, [j], [1]) + tf.slice(u_r_ewm_h_previous, [j], [1]))

            hj = zj*tf.slice(ht_previous, [j], [1]) + (1-zj)*hj_tilde
            ht = ht + tf.sparse_tensor_to_dense(tf.SparseTensor([j], [hj], [1000]))

        ht_previous = ht

    c = tf.tanh(tf.matmul(v, ht_previous))