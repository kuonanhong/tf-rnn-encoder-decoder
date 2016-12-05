from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections
import zipfile
import re
from six.moves import range

k = 150
num_batches = 1000

def read_data(filename):
    z = zipfile.ZipFile(filename, 'r')
    lines = list()
    line_count = 0
    with z.open(z.namelist()[0]) as f:
        for line in f:
            line_count += 1
            line = line.decode('utf-8')
            lines.append(line.replace(".", "").replace("?", "").replace("!", ""))
            if line_count == num_batches:
                break
    z.close()
    return lines

french = read_data('fr.zip')
english = read_data('en.zip')
print('Data size %d %d' % (len(english), len(french)))

def build_dataset(sentences):
  count = [['UNK', -1]]
  count.extend(collections.Counter(re.findall(r'\w+', ' '.join(sentences).lower())).most_common(k - 1))
  indeces = dict()
  for word, _ in count:
    indeces[word] = len(indeces)
  data = list()
  unk_count = 0
  for sentence in sentences:
    words = re.findall(r'[\w]+', sentence.lower())
    sentence_arr = np.zeros((k, 0))
    for word in words:
        if word in indeces:
            index = indeces[word]
        else:
            index = 0
            unk_count += 1
        vector = np.zeros((k, 1))
        vector[index] = 1
        sentence_arr = np.concatenate((sentence_arr, vector), axis=1)
    data.append(sentence_arr)
  count[0][1] = unk_count
  return data, count, indeces

fr_data, fr_count, fr_indeces = build_dataset(french)
print('Most common words (+UNK)', fr_count[:5])
en_data, en_count, en_indeces = build_dataset(english)
print('Most common words (+UNK)', en_count[:5])
del english, french  # Hint to reduce memory.

dim_embedding = 50
dim_hidden = 100

graph = tf.Graph()
with graph.as_default():
    # Variables
    # Encoder embedding
    we = tf.Variable(tf.truncated_normal([dim_embedding, k], -0.1, 0.1))
    # Encoder update gate
    wz = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    uz = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Encoder reset gate
    wr = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    ur = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Encoder h~ [find name]
    w = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    u = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Encoder representation weight
    v = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Decoder representation weight
    v_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Decoder embedding
    w_prime_e = tf.Variable(tf.truncated_normal([dim_embedding, k], -0.1, 0.1))
    # Decoder update gate
    w_prime_z = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    u_prime_z = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    Cz = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Decoder reset gate
    w_prime_r = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    u_prime_r = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    Cr = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Decoder h~ [find name]
    w_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_embedding], -0.1, 0.1))
    u_prime = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    C = tf.Variable(tf.truncated_normal([dim_hidden, dim_hidden], -0.1, 0.1))
    # Decoder maxout calculation
    oh = tf.Variable(tf.truncated_normal([2 * dim_hidden, dim_hidden], -0.1, 0, 1))
    oy = tf.Variable(tf.truncated_normal([2 * dim_hidden, k], -0.1, 0, 1))
    oc = tf.Variable(tf.truncated_normal([2 * dim_hidden, dim_hidden], -0.1, 0, 1))
    # Decoder output
    gl = tf.Variable(tf.truncated_normal([k, dim_embedding], -0.1, 0, 1))
    gr = tf.Variable(tf.truncated_normal([dim_embedding, dim_hidden], -0.1, 0, 1))

    # Encoder
    # TODO: h_previous = tf.zeros([k, 1])
    # TODO: xt = tf.reshape(X[:, t], [k, 1])
    h_previous = tf.placeholder(tf.float32, shape=[dim_hidden, 1])
    xt = tf.placeholder(tf.float32, shape=[k, 1])
    # Current vector and its embedding
    e = tf.matmul(we, xt)
    # Reset calculation
    r = tf.sigmoid(tf.matmul(wr, e) + tf.matmul(ur, h_previous))
    # Update calculation
    z = tf.sigmoid(tf.matmul(wz, e) + tf.matmul(uz, h_previous))
    # Hidden-tilde calculation
    h_tilde = tf.tanh(tf.matmul(w, e) + tf.matmul(u, r * h_previous))
    # Hidden calculation
    one = tf.ones([dim_hidden, 1])
    h = z * h_previous + (one - z) * h_tilde

    # Summary calculation
    c = tf.tanh(tf.matmul(v, h_previous))
    h_prime_init = tf.tanh(tf.matmul(v_prime, c))

    # Decoder
    # TODO: y_previous = tf.zeros([k, 1])
    # TODO: h_prime_previous = tf.tanh(tf.matmul(v_prime, c))
    y_previous = tf.placeholder(tf.float32, shape=[k, 1])
    h_prime_previous = tf.placeholder(tf.float32, shape=[dim_hidden, 1])
    # Current vector's embedding
    e = tf.matmul(w_prime_e, y_previous)
    # Reset calculation
    r_prime = tf.sigmoid(tf.matmul(w_prime_r, e) + tf.matmul(u_prime_r, h_prime_previous) + tf.matmul(Cr, c))
    # Update calculation
    z_prime = tf.sigmoid(tf.matmul(w_prime_z, e) + tf.matmul(u_prime_z, h_prime_previous) + tf.matmul(Cz, c))
    # Hidden-tilde calculation
    h_tilde_prime = tf.tanh(tf.matmul(w_prime, e) + r_prime * (tf.matmul(u_prime, h_prime_previous) + tf.matmul(C, c)))
    # Hidden calculation
    one = tf.ones([dim_hidden, 1])
    h_prime = z_prime * h_prime_previous + (one - z_prime) * h_tilde_prime
    # Maxout calculation
    s_prime = tf.matmul(oh, h_prime) + tf.matmul(oy, y_previous) + tf.matmul(oc, c)
    s = tf.reshape(tf.reduce_max(tf.reshape(s_prime, [dim_hidden, 2]), 1), [dim_hidden, 1])
    # Logit calculation
    g = tf.matmul(gl, gr)
    logit = tf.matmul(g, s)

    # Optimizer
    logits = tf.placeholder(tf.float32, shape=[k, None])
    Y = tf.placeholder(tf.float32, shape=[k, None])
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # TODO: figure out what's going wrong
    tf.add(global_step, 1)

sess = tf.Session(graph=graph)

with sess:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_batches):
        # Get input and target matrices
        input = en_data.pop()
        target = fr_data.pop()

        # Run encoder
        h_prev = np.zeros([dim_hidden, 1])
        for t in range(input.shape[1]):
            h_prev = sess.run([h], feed_dict={h_previous: h_prev, xt: input[:, t].reshape((k, 1))})
            h_prev = np.asarray(h_prev).reshape((dim_hidden, 1))

        # Run decoder
        h_prime_prev = sess.run([h_prime_init], feed_dict={h_previous: h_prev})
        y_prev = np.zeros([k, 1])
        logit_list = list()
        for t in range(target.shape[1]):
            h_prime_prev = np.asarray(h_prime_prev).reshape((dim_hidden, 1))
            print("Shape: %s; Dtype: %s" % (h_prime_prev.shape, h_prime_prev.dtype))
            # TODO: figure out what's going wrong
            h_prime_prev, logit_current = sess.run([h_prime, logit], feed_dict={y_previous: y_prev, h_prime_previous: h_prime_prev})
            print("test")
            logit_current = np.asarray(logit_current).reshape(k, 1)
            y_prev = target[:, t].reshape((k, 1))
            logit_list.append(logit_current)

        # Run optimization
        _, l = sess.run([optimizer, loss], feed_dict={logits: logit_list, Y: target})
        if (step+1) % 100 == 0:
            print("Loss at step %d: %f" % (step, l))