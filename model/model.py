# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: model.py
# Python  : python3.6
# Time    : 18-7-25 21:45

import tensorflow as tf

def rnn_model(input_data, output_data, voc_size=6110, hidden_size=100, num_layers=2, batch_size=100,
              learning_rate=0.01, LSTM_KEEP_PROB = 0.9, EMBEDDING_KEEP_PROB = 0.9):
    """
    :param input: shape -- [batch_size, num_steps]
    :param output: shape -- [batch_size * num_steps, voc_size]
    :param voc_size:
    :param hidden_size:
    :param num_layers:
    :param learning_rate:
    :param LSTM_KEEP_PROB: lstm keep prob
    :param EMBEDDING_KEEP_PROB: embedding keep prob
    :return:
    """
    params = dict()
    if output_data is None: # "Using a `tf.Tensor` as a Python `bool` is not allowed. "
        LSTM_KEEP_PROB, EMBEDDING_KEEP_PROB = 1, 1
    lstm_cells = [
        tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
            output_keep_prob=LSTM_KEEP_PROB
        ) for _ in range(num_layers)
    ]
    cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
    if output_data is not None:
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
    else: # forward时输入为1
        init_state = cell.zero_state(1, tf.float32)
    embedding = tf.get_variable('embedding', shape=[voc_size, hidden_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    # inputs shape: [batch_size, num_steps, hidden_size]
    inputs = tf.nn.embedding_lookup(embedding, input_data, name='lstm_input')
    # 只在训练时使用dropout
    if output_data is not None:
        inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

    # outputs shape: [batch_size, num_steps, hidden_size]
    # final_state shape: [batch_size, hidden_size]
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state) # shape: [batch_size, num_steps,hidden_size]
    outputs = tf.reshape(outputs, [-1, hidden_size]) # shape: [batch_size*num_steps, hidden_size]

    # weights复用embedding变量?
    # weights = tf.get_variable('weights', shape=[hidden_size, voc_size],
    #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    weights = tf.transpose(embedding, name='weights')
    bias = tf.get_variable('bias', [voc_size], initializer=tf.constant_initializer(0.0))
    logits = tf.add(tf.matmul(outputs, weights), bias, name='logits')

    if output_data is not None:
        # 定义交叉熵损失函数和平均损失函数
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(output_data, [-1]),  # 将[batch_size, num_steps] reshape为[batch_size*num_steps]
            logits=logits
        ) #todo:补全部分的损失值怎么处理？
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        params['initial_state'] = init_state
        params['output'] = outputs
        params['train_op'] = train_op
        params['total_loss'] = total_loss
        params['loss'] = loss
        params['last_state'] = final_state

    else:
        prediction = tf.nn.softmax(logits)
        params['initial_state'] = init_state
        params['last_state'] = final_state
        params['prediction'] = prediction
    return params



