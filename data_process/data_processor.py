# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: data_processor.py
# Python  : python3.6
# Time    : 18-7-24 22:17

import os
import json
import numpy as np

begin = 'B'
end = 'E'
root_path = os.path.dirname(os.path.dirname(__file__))

def load_transfer():
    with open(os.path.join(root_path, 'data/word2num'), 'r') as f:
        word2num = json.load(f)
    with open(os.path.join(root_path, 'data/num2word'), 'r') as f:
        num2word = json.load(f)
    return word2num, num2word

def gen_dataset():
    word_dict = dict()
    content_list = []
    with open(os.path.join(root_path, 'data/poems.txt'), 'r') as f:
        for l in f:
            try:
                title, content = l.split(":")
            except Exception as e:
                continue
            content = content.replace(' ', '').replace('\n', '') # 去除空格及换行符
            if set('()（））_《》[]') & set(content):
                continue
            if begin in content or end in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            # 必须要加入start, 因为输入的一般为第一个词
            # 应使rnn习得第一个词的状态
            content = begin + content + end
            for c in content: # 每一个字符(包括标点)作为一个token
                word_dict[c] = word_dict.get(c, 0) + 1
            content_list.append(content)
    # word_dict["<end>"] = -1
    # word_dict["<start>"] = -2
    word_dict[' '] = -1
    print(len(word_dict))
    voc = sorted(word_dict.items(), key=lambda x: x[1])
    word2num = {voc[i][0]: i for i in range(len(voc))}
    num2word = {i: voc[i][0] for i in range(len(voc))}
    with open(os.path.join(root_path, 'data/word2num'), 'w') as f:
        json.dump(word2num, f, ensure_ascii=False, indent=2)
    with open(os.path.join(root_path, 'data/num2word'), 'w') as f:
        json.dump(num2word, f, ensure_ascii=False, indent=2)

    # generate train files
    f_out = open(os.path.join(root_path, 'data/train.data'), 'w+')
    for cl in content_list:
        nums = [word2num[c] for c in cl]
        nums = list(map(str, nums))
        f_out.write(' '.join(nums) + '\n')

def make_batch(BATCH_SIZE = 60): # 固定?
    """有两种构造数据集的方式"""
    """一种不重叠，另一种重叠"""
    id_list = []
    word2num, _ = load_transfer()
    with open(os.path.join(root_path, 'data/train.data'), 'r') as f: # 对所有数据进行分割
        for l in f:
            li = list(map(int, l.strip().split()))
            id_list.append(li) # 完整的一首诗作为一个sample
    x_batches, y_batches = [], []
    batch_num = len(id_list) // BATCH_SIZE
    for i in range(batch_num):
        data = id_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        num_step = max(map(len, data)) # 取每一首诗最长一句作为num_step
        x = np.full((BATCH_SIZE, num_step), word2num[' '], np.int32) # 全部以end填充的数据模板
        for i, sample in enumerate(data):
            x[i, :len(sample)] = sample
        y = np.copy(x)
        y[:,:-1] = x[:,1:] # x: a b c end; y: b c end end; end -> end
        x_batches.append(x)
        y_batches.append(y)
    return x_batches, y_batches

if __name__ == '__main__':
    gen_dataset()
    x, y = make_batch()
    print(len(x), len(y))
    print(x[0][1])
    print(y[0][1])
