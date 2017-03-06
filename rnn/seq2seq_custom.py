from tensorflow.python.framework import dtypes
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.python.ops import variable_scope
import tensorflow.contrib.legacy_seq2seq as seq2seq
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

dim_hidden = 100

def infer(initial_state,
          cell,
          num_units,
          scope=None):


    with variable_scope.variable_scope(scope or "infer"):
        state = initial_state
        outputs = []
        prev = None
        for i in range(10):
            inp = tf.zeros([num_units, num_units], tf.float32)
            if i > 0:
                inp = prev
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            prev = output
    return outputs, state

def custom_rnn_seq2seq(encoder_inputs,
                      decoder_inputs,
                      enc_cell,
                      dec_cell,
                      dtype=dtypes.float32,
                      initial_state=None,
                      use_previous=False,
                      scope=None,
                      num_units=0):

    with variable_scope.variable_scope(scope or "custom_rnn_seq2seq"):
        _, enc_state = core_rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype, scope=scope, initial_state=initial_state)
        print(enc_state.get_shape)
        c = tf.tanh(tf.matmul(tf.get_variable("v", [dim_hidden, dim_hidden]), enc_state))
        h_prime_init = tf.tanh(tf.matmul(tf.get_variable("v_prime", [dim_hidden, dim_hidden]), c))
        if not use_previous:
            return seq2seq.rnn_decoder(decoder_inputs, LSTMStateTuple(c, h_prime_init), dec_cell, scope=scope)
        return infer(LSTMStateTuple(c, h_prime_init), dec_cell, num_units)