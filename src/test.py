import numpy as np
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq
import tensorflow.contrib.rnn as rnn_cell



seq_length = 5
batch_size = 64

k = 150
embedding_dim = 50

memory_dim = 100

graph = tf.Graph()
with graph.as_default():
    enc_inp = [tf.placeholder(tf.float32, shape=(k, 1), name="inp%i" % t)
               for t in range(seq_length)]

    labels = [tf.placeholder(tf.int32, shape=(k, 1), name="labels%i" % t)
              for t in range(seq_length)]

    weights = [tf.ones_like(labels_t, dtype=tf.float32)
               for labels_t in range(k)]

    # Decoder input: prepend some "GO" token and drop the final
    # token of the encoder input
    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")]
    for t in range(len(labels) - 1):
        dec_inp.append(tf.cast(labels[t], tf.float32))


    # Initial memory value for recurrence.
    prev_mem = tf.zeros((batch_size, memory_dim))

    cell = rnn_cell.GRUCell(memory_dim)

    dec_outputs, dec_memory = seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, cell)

    loss = seq2seq.sequence_loss(dec_outputs, labels, weights, k)

    # Optimizer
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

sess = tf.Session(graph=graph)
with sess:
    def train_batch(batch_size):
        X = [np.random.choice(k, size=(seq_length,), replace=False)
             for _ in range(batch_size)]
        Y = X[:]

        # Dimshuffle to seq_len * batch_size
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

        _, loss_t = sess.run([optimizer, loss], feed_dict)
        return loss_t

    tf.global_variables_initializer().run()
    for step in range(500):
        l = train_batch(batch_size)
        if (step + 1) % 100 == 0:
            print("Loss at step %d: %f" % (step, l))