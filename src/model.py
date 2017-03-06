from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
import tensorflow.contrib.legacy_seq2seq as seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell
from rnn import seq2seq_custom
from rnn.encoder_decoder_cell import EncoderCell
from rnn.encoder_decoder_cell import DecoderCell
import pickle

k = 150
num_batches = 1000

dim_embedding = 50
dim_hidden = 100
seq_length = 5

with tf.variable_scope("encoder_decoder_scope") as encoder_decoder_scope:
    # Encoder embedding
    we = tf.get_variable("we", [dim_embedding, k], initializer=tf.random_normal_initializer())
    # # Encoder update gate
    wz = tf.get_variable("wz", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    uz = tf.get_variable("uz", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Encoder reset gate
    wr = tf.get_variable("wr", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    ur = tf.get_variable("ur", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Encoder h~ [find name]
    w = tf.get_variable("w", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    u = tf.get_variable("u", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Encoder representation weight
    v = tf.get_variable("v", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())

    # Decoder representation weight
    v_prime = tf.get_variable("v_prime", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Decoder embedding
    w_prime_e = tf.get_variable("w_prime_e", [dim_embedding, k], initializer=tf.random_normal_initializer())
    # Decoder update gate
    w_prime_z = tf.get_variable("w_prime_z", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    u_prime_z = tf.get_variable("u_prime_z", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    Cz = tf.get_variable("Cz", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Decoder reset gate
    w_prime_r = tf.get_variable("w_prime_r", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    u_prime_r = tf.get_variable("u_prime_r", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    Cr = tf.get_variable("Cr", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Decoder h~ [find name]
    w_prime = tf.get_variable("w_prime", [dim_hidden, dim_embedding], initializer=tf.random_normal_initializer())
    u_prime = tf.get_variable("u_prime", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    C = tf.get_variable("C", [dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Decoder maxout calculation
    oh = tf.get_variable("oh", [2 * dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    oy = tf.get_variable("oy", [2 * dim_hidden, k], initializer=tf.random_normal_initializer())
    oc = tf.get_variable("oc", [2 * dim_hidden, dim_hidden], initializer=tf.random_normal_initializer())
    # Decoder output
    gl = tf.get_variable("gl", [k, dim_embedding], initializer=tf.random_normal_initializer())
    gr = tf.get_variable("gr", [dim_embedding, dim_hidden], initializer=tf.random_normal_initializer())

fr_data = pickle.load(open("../data/fr_data.p", "rb"))
en_data = pickle.load(open("../data/en_data.p", "rb"))

graph = tf.Graph()
with graph.as_default():
    enc_inp = [tf.placeholder(tf.float32, shape=(k, 1), name="inp%i" % t)
               for t in range(seq_length)]

    labels = [tf.placeholder(tf.int32, shape=(k, 1), name="labels%i" % t)
              for t in range(seq_length)]

    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")]
    for t in range(len(labels) - 1):
        dec_inp.append(tf.cast(labels[t], tf.float32))

    initial_state = tf.zeros([100, 1])

    # Initial memory value for recurrence.
    enc_cell = EncoderCell(dim_hidden)
    dec_cell = DecoderCell(dim_hidden)

    dec_outputs, dec_state = seq2seq_custom.custom_rnn_seq2seq(enc_inp, dec_inp, enc_cell, dec_cell,
                                                                scope=encoder_decoder_scope, initial_state=initial_state)

    log_perp_list = []
    for logit, target in zip(dec_outputs, labels):
        log_perp_list.append(tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                                     logits=tf.clip_by_value(logit, 1e-12, 1.0)))
    log_perps = tf.add_n(log_perp_list)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=(dec_outputs+ 1e-12)))
    loss = tf.reduce_sum(log_perps)/5

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