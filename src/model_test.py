import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell, AttentionCellWrapper
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
import pprint
import numpy as np
def load_w2v():
    return
class lstm_att_model():
    def __init__(self,word_embedding_size,voc_size,hidden_size,input_size,w2v_flag = False):
        self.word_embedding_size = word_embedding_size
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.input = tf.placeholder(dtype=tf.int64, shape=[None, input_size])
        self.seq_len = tf.placeholder(tf.int64, [None], name='seq_len')
        self.target = tf.placeholder(dtype=tf.int64, shape=[None], name='target')
        if w2v_flag is False:
            self.word_embedding = tf.Variable(tf.random_uniform(shape=[voc_size, word_embedding_size]),trainabel = "True")
        else:
            self.word_embedding = tf.Variable(load_w2v())
        output_bilstm = self.bilstm(self.word_embedding,self.hidden_size,self.seq_len)
        att_output = self.attention(output_bilstm,word_embedding_size)

        #全链接层


    def bilstm(self,inputs,hid_size,seq_len):
        fw_cell = LSTMCell(hid_size)
        bw_cell = LSTMCell(hid_size)
        outputs ,final_state = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, seq_len,dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        final_state = tf.concat(final_state, 1)
        return outputs,final_state

    def attention(inputs, attention_size):
        """
        Attention mechanism layer.

        :param inputs: outputs of RNN/Bi-RNN layer (not final state)
        :param attention_size: linear size of attention weights
        :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
        """
        # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        return output

