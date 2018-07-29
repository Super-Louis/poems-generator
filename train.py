# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: train.py
# Python  : python3.6
# Time    : 18-7-25 23:08
import os
# set tensorflow logging level: [DEBUG, INFO, WARN, ERROR, FATAL]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from model.model import rnn_model
from data_process.data_processor import make_batch

EPOCHES = 10
BATCH_SIZE = 60
VOC_SIZE = 6110 # 根据processed data得到

def train():
    input_batches, out_batches = make_batch(BATCH_SIZE)
    input = tf.placeholder(shape=[BATCH_SIZE, None], name='input', dtype=tf.int32) # int for embedding look up
    label = tf.placeholder(shape=[BATCH_SIZE, None], name='label', dtype=tf.int32)
    train_params = rnn_model(input, label, VOC_SIZE, batch_size=BATCH_SIZE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        try:
            for ep in range(EPOCHES):
                count = 0
                for in_batch, out_batch in zip(input_batches, out_batches):
                    loss, _, _ = sess.run([
                        train_params['total_loss'],
                        train_params['last_state'],
                        train_params['train_op']
                    ], feed_dict={input: in_batch, label: out_batch})
                    count += 1
                    if count % 100 == 0:
                        print("after %d epoch and %d batches, loss: %.3f" % (ep, count, loss))
        except KeyboardInterrupt:
            pass
        finally:
            saver.save(sess, 'model/poem_model.ckpt')

if __name__ == '__main__':
    train()

