# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: generate_poems.py
# Python  : python3.6
# Time    : 18-7-27 21:18

import tensorflow as tf
import numpy as np
from model.model import rnn_model
from data_process.data_processor import load_transfer

word2num, num2word = load_transfer()

def random_word(prediction):
    """probs to word"""
    t = np.cumsum(prediction)  # prefix sum
    s = np.sum(prediction)
    coff = np.random.rand(1)
    index = int(np.searchsorted(t, coff * s))
    return index

def generate_with_head(ini_word):
    poems = ''
    input = tf.placeholder(shape=[None, None], name='input', dtype=tf.int32)
    train_params = rnn_model(input, None)
    saver = tf.train.Saver()
    word = ini_word
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
        input_data = [[word2num['B']]]
        prediction, last_state, _ = sess.run([train_params['prediction'],
                                              train_params['last_state'],
                                              train_params['initial_state']],
                                             feed_dict={input: input_data})
        while word != 'E':
            poems += word
            input_data = [[word2num[str(word)]]]
            prediction, last_state, _ = sess.run([train_params['prediction'],
                                         train_params['last_state'],
                                         train_params['initial_state']],
                                        feed_dict={input: input_data, train_params['initial_state']: last_state})
            pred = random_word(prediction[0])
            word = num2word[str(pred)]
            print(word)
        # if len(poems) != 32:
        #     return generate_with_head(ini_word)
    return poems

if __name__ == '__main__':
    print(generate_with_head('春'))


