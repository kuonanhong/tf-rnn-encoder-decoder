from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import zipfile
import re
from six.moves import range
import tensorflow.contrib.legacy_seq2seq as seq2seq
import tensorflow.contrib.rnn as rnn_cell
import seq2seq_custom
import pickle

k = 150
num_batches = 1000
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

dim_embedding = 50
dim_hidden = 100
seq_length = 5
memory_dim = 100

# graph = tf.Graph()
# def unroll(input, target):
#     _, loss = seq2seq.model_with_buckets(input, target, target[1:], weights, buckets, seq2seq.basic_seq2seq)


    # Variables
    # Encoder embedding
    # we = tf.Variable(tf.truncated_normal([dim_embedding, k], -0.1, 0.1))
    # # Encoder update gate
    # wz = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # uz = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Encoder reset gate
    # wr = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # ur = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Encoder h~ [find name]
    # w = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # u = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Encoder representation weight
    # v = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Decoder representation weight
    # v_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Decoder embedding
    # w_prime_e = tf.Variable(tf.truncated_normal([dim_embedding, k], -0.1, 0.1))
    # # Decoder update gate
    # w_prime_z = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # u_prime_z = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Cz = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Decoder reset gate
    # w_prime_r = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # u_prime_r = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Cr = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Decoder h~ [find name]
    # w_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    # u_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # C = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # # Decoder maxout calculation
    # oh = tf.Variable(tf.truncated_normal([2 * dim_hidden, dim_hidden], -0.1, 0, 1))
    # oy = tf.Variable(tf.truncated_normal([2 * dim_hidden, k], -0.1, 0, 1))
    # oc = tf.Variable(tf.truncated_normal([2 * dim_hidden, dim_hidden], -0.1, 0, 1))
    # # Decoder output
    # gl = tf.Variable(tf.truncated_normal([k, dim_embedding], -0.1, 0, 1))
    # gr = tf.Variable(tf.truncated_normal([dim_embedding, dim_hidden], -0.1, 0, 1))
    #
    # # Encoder
    # saved_h_previous = tf.Variable(tf.zeros([dim_hidden, 1]), trainable=False)
    # h_previous = saved_h_previous
    # xt = tf.placeholder(tf.float32, shape=[k, 1])
    #
    # # Current vector and its embedding
    # e = tf.matmul(we, xt)
    # # Reset calculation
    # r = tf.sigmoid(tf.matmul(wr, e) + tf.matmul(ur, h_previous))
    # # Update calculation
    # z = tf.sigmoid(tf.matmul(wz, e) + tf.matmul(uz, h_previous))
    # # Hidden-tilde calculation
    # h_tilde = tf.tanh(tf.matmul(w, e) + tf.matmul(u, r * h_previous))
    # # Hidden calculation
    # one = tf.ones([dim_hidden, 1])
    # h = z * h_previous + (one - z) * h_tilde
    #
    # # Summary calculation
    # with tf.control_dependencies([saved_h_previous.assign(h)]):
    #     c = tf.tanh(tf.matmul(v, saved_h_previous))
    #     h_prime_init = tf.tanh(tf.matmul(v_prime, c))
    #
    # # Decoder
    # with tf.control_dependencies([h_prime_init]):
    #     saved_h_prime_previous = tf.Variable(h_prime_init, trainable=False)
    # saved_y_previous = tf.Variable(tf.zeros([k, 1]), trainable=False)
    # y_previous = saved_y_previous
    # h_prime_previous = saved_h_prime_previous
    # logits = tf.Variable(tf.zeros([k, None], dtype=tf.float32), trainable=False)
    #
    # # Current vector's embedding
    # e = tf.matmul(w_prime_e, y_previous)
    # # Reset calculation
    # r_prime = tf.sigmoid(tf.matmul(w_prime_r, e) + tf.matmul(u_prime_r, h_prime_previous) + tf.matmul(Cr, c))
    # # Update calculation
    # z_prime = tf.sigmoid(tf.matmul(w_prime_z, e) + tf.matmul(u_prime_z, h_prime_previous) + tf.matmul(Cz, c))
    # # Hidden-tilde calculation
    # h_tilde_prime = tf.tanh(tf.matmul(w_prime, e) + r_prime * (tf.matmul(u_prime, h_prime_previous) + tf.matmul(C, c)))
    # # Hidden calculation
    # one = tf.ones([dim_hidden, 1])
    # h_prime = z_prime * h_prime_previous + (one - z_prime) * h_tilde_prime
    # # Maxout calculation
    # s_prime = tf.matmul(oh, h_prime) + tf.matmul(oy, y_previous) + tf.matmul(oc, c)
    # s = tf.reshape(tf.reduce_max(tf.reshape(s_prime, [dim_hidden, 2]), 1), [dim_hidden, 1])
    # # Logit calculation
    # g = tf.matmul(gl, gr)
    # logit = tf.matmul(g, s)
    # print(logits.get_shape())
    # print(logit.get_shape())
    # logits.assign(tf.concat(1, [logits, logit]))
    #
    # # Classifier
    # Y = tf.placeholder(tf.float32, shape=[k, None])
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
    # # global_step = tf.Variable(0)
    # # learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    #
    # # Optimizer
    # learning_rate = 0.1
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # # tf.add(global_step, 1)

fr_data = pickle.load(open("../data/fr_data.p", "rb"))
en_data = pickle.load(open("../data/en_data.p", "rb"))

graph = tf.Graph()
with graph.as_default():
    enc_inp = [tf.placeholder(tf.float32, shape=(k, 1), name="inp%i" % t)
               for t in range(seq_length)]

    labels = [tf.placeholder(tf.int32, shape=(k, 1), name="labels%i" % t)
              for t in range(seq_length)]

    weights = [tf.ones_like(labels_t, dtype=tf.float32)
               for labels_t in range(seq_length)]

    # Decoder input: prepend some "GO" token and drop the final
    # token of the encoder input
    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")]
    for t in range(len(labels) - 1):
        dec_inp.append(tf.cast(labels[t], tf.float32))


    # Initial memory value for recurrence.
    enc_cell = rnn_cell.BasicLSTMCell(memory_dim)
    dec_cell = rnn_cell.BasicLSTMCell(memory_dim)

    dec_outputs, dec_memory = seq2seq_custom.custom_rnn_seq2seq(enc_inp, dec_inp, enc_cell, dec_cell)

    loss = seq2seq.sequence_loss(dec_outputs, labels, weights, k)

    # Optimizer
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

sess = tf.Session(graph=graph)
with sess:
    def train_batch(X, Y):
        if X.shape[1] < seq_length:
            for _ in range(seq_length - X.shape[1]):
                X = np.concatenate((X, np.zeros((150, 1))), axis=1)

        if Y.shape[1] < seq_length:
            for _ in range(seq_length - Y.shape[1]):
                Y = np.concatenate((Y, np.zeros((150, 1))), axis=1)

        feed_dict = {enc_inp[t]: X[:, t].reshape(150, 1) for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[:, t].reshape(150, 1) for t in range(seq_length)})

        _, loss_t = sess.run([optimizer, loss], feed_dict)
        return loss_t


    tf.global_variables_initializer().run()
    for step in range(num_batches):
        input = en_data.pop()
        target = fr_data.pop()
        l = train_batch(input, target)
        if (step + 1) % 100 == 0:
            print("Loss at step %d: %f" % (step, l))
    # tf.initialize_all_variables().run()
    # print("Initialized")
    # for step in range(num_batches):
    #     # # Get input and target matrices
    #     input = en_data.pop()
    #     target = fr_data.pop()
    #     output, _ = seq2seq.model_with_buckets(input, target, target[1:], [], buckets, seq2seq.basic_rnn_seq2seq)
    #     loss = tf.nn.softmax_cross_entropy_with_logits(target[:1], output)
    #     learning_rate = 0.1
    #     train = optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #     #
    #     # # Run encoder
    #     # for t in range(input.shape[1]):
    #     #     _ = sess.run([h], feed_dict={xt: input[:, t].reshape((k, 1))})
    #     #
    #     # # Compute summary calculation
    #     # _ = sess.run([h_prime_init])
    #     #
    #     # # Run decoder
    #     # for t in range(target.shape[1]):
    #     #     _ = sess.run([logits])
    #     #
    #     # # Run optimization
    #     # _, l = sess.run([optimizer, loss], feed_dict={Y: target})
    #     if (step+1) % 100 == 0:
    #         print("Loss at step %d: %f" % (step, loss))