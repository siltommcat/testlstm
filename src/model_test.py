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
