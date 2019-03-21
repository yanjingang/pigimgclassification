#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: read_cifar10.py
Desc: 读取cifar10数据集的图片和label，并生成以序号+label命名的图片
Author:yanjingang(yanjingang@mail.com)
Date: 2018/12/25 23:12
Cmd: nohup python read_cifar10.py >log/read_cifar10.log &
"""

import os
import sys
from PIL import Image
import cPickle

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)
# print(CUR_PATH, BASE_PATH)
from machinelearning.lib import utils

# cifar10训练集目录  from: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cifar10_path = CUR_PATH + '/data/cifar-10-batches-py/'
train_batchs = [
    cifar10_path + 'data_batch_1',
    cifar10_path + 'data_batch_2',
    cifar10_path + 'data_batch_3',
    cifar10_path + 'data_batch_4',
    cifar10_path + 'data_batch_5'
]
test_batchs = [cifar10_path + 'test_batch']

# 读取出的图片存放位置
output_path = CUR_PATH + '/data/'

# label含义
label_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def reader_cifar10(batchs, path='train'):
    """读取cifar10数据集的图片和label，并生成以序号-label-labelname命名的图片"""
    id = 0  # 图片集序号
    for file in batchs:
        data = {}
        with open(file, 'rb') as fo:
            data = cPickle.load(fo)
            print(data.keys())
            # print(data['data'])
        for i in xrange(len(data['data'])):  # 遍历图片像素数据
            id += 1
            # 读取单张图片数据
            img = data['data'][i]
            label = data['labels'][i]
            label_name = label_list[label]
            print(img)
            print(len(img))
            print(str(label) + ' : ' + label_name)

            # 重建rgb彩色图片
            img = img.reshape(3, 32, 32)
            print(img)
            print(img.shape)
            r = Image.fromarray(img[0]).convert('L')
            g = Image.fromarray(img[1]).convert('L')
            b = Image.fromarray(img[2]).convert('L')
            new_img = Image.merge('RGB', (r, g, b))

            # 保存图片(序号-label-labelname.png)
            utils.mkdir(output_path + path)
            save_file = output_path + path + '/' + str(id) + '-' + str(label) + '-' + label_name + '.png'
            new_img.save(save_file)
            print save_file
            # if id > 10:
            #    break
        # break


if __name__ == '__main__':
    # 读取训练集
    # reader_cifar10(train_batchs, path='train')
    # 读取测试集
    reader_cifar10(test_batchs, path='test')
