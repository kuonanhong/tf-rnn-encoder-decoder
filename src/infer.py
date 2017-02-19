import tensorflow as tf
import seq2seq_custom
import tensorflow.contrib.rnn as rnn_cell
import pickle
import numpy as np

seq_length = 4
k = 150
memory_dim = 100
en_indeces = pickle.load(open("../data/en_indeces.p", "rb"))
en_tokens = {v:k for k,v in en_indeces.items()}

fr_indeces = pickle.load(open("../data/fr_indeces.p", "rb"))
fr_tokens = {v:k for k,v in fr_indeces.items()}

def get_one_hot(input):
    one_hots = []
    for _, token in enumerate(input.split()):
        print(en_indeces[token])
        vector = np.zeros((k, 1), np.float32)
        vector[en_indeces[token]] = 1
        one_hots.append(vector)
    return one_hots

def get_tokens(output):
    sentence = ""
    for t in range(output.__len__()):
        vector = output[t]
        for i in range(vector.shape[0]):
            if vector[i] == 1:
                sentence += fr_tokens[i] + ' '
    return sentence

graph = tf.Graph()
with graph.as_default():
    enc_inp = [tf.placeholder(tf.float32, shape=(k, 1), name="inp%i" % t)
               for t in range(seq_length)]

    enc_cell = rnn_cell.BasicLSTMCell(k)
    dec_cell = rnn_cell.BasicLSTMCell(k)

    inference, _ = seq2seq_custom.custom_rnn_seq2seq(enc_inp, None, enc_cell, dec_cell, use_previous=True, num_units=k)
    print(tf.shape(inference))

sess = tf.Session(graph=graph)
with sess:
    tf.global_variables_initializer().run()
    sentence = input('Enter your input: ')
    inp = get_one_hot(sentence)
    feed_dict = {enc_inp[t]: inp[t] for t in range(seq_length)}
    inf = sess.run([inference], feed_dict=feed_dict)
    #prediction = get_tokens(inf)