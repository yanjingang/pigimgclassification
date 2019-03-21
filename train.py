#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: train.py
Desc: 图像分类模型（resnet|vgg）
Author:yanjingang(yanjingang@mail.com)
Date: 2018/12/26 23:34
Cmd: python train.py
"""

from __future__ import print_function
import sys
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.trainer import *
from paddle.fluid.contrib.inferencer import *
import numpy

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../../')
sys.path.append(BASE_PATH)
# print(CUR_PATH, BASE_PATH)
from machinelearning.lib.resnet import resnet_cifar10
from machinelearning.lib import utils


def image_classification_network():
    """定义图像分类输入层及网络结构: resnet or vgg"""
    # 输入层：The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='img', shape=data_shape, dtype='float32')

    # 网络模型
    # resnet
    predict = resnet_cifar10(images, 32)
    # vgg
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict


def train_network():
    """定义训练输入层、网络结果、label数据层、损失函数等训练参数"""
    # 定义输入img层及网络结构resnet
    predict = image_classification_network()
    # 定义训练用label数据层
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # 定义训练损失函数cost
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    # accuracy用于在迭代过程中print
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]


def optimizer_program():
    """定义优化器"""
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, params_dirname="model"):
    """开始训练"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    BATCH_SIZE = 128
    EPOCH_NUM = 5

    # 定义训练和测试数据batch reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            # paddle.dataset.cifar.train10()
            utils.image_reader_creator(CUR_PATH + '/data/train/', 32, 32, rgb=True, reshape1=True)  # 自己读取images
            , buf_size=50000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        # paddle.dataset.cifar.test10()
        utils.image_reader_creator(CUR_PATH + '/data/test/', 32, 32, rgb=True, reshape1=True)  # 自己读取images
        , batch_size=BATCH_SIZE)

    # 定义event_handler，输出训练过程中的结果
    lists = []

    def event_handler(event):
        if isinstance(event, EndStepEvent):
            if event.step % 100 == 0:
                print("\nPass %d, Batch %d, Cost %f, Acc %f" %
                      (event.step, event.epoch, event.metrics[0],
                       event.metrics[1]))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

        if isinstance(event, EndEpochEvent):
            avg_cost, accuracy = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])

            print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                event.epoch, avg_cost, accuracy))
            if params_dirname is not None:
                trainer.save_params(params_dirname)
            # 保存训练结果损失情况
            lists.append((event.epoch, avg_cost, accuracy))

    # 创建训练器(train_func损失函数; place是否使用gpu; optimizer_func优化器)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    trainer = Trainer(train_func=train_network, optimizer_func=optimizer_program, place=place)

    # 开始训练模型
    trainer.train(
        reader=train_reader,
        num_epochs=EPOCH_NUM,
        event_handler=event_handler,
        feed_order=['img', 'label'])

    # 找到训练误差最小的一次结果(trainer.save_params()自动做了最优选择，这里只是为了验证EPOCH_NUM设置几次比较合理)
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))


def infer(use_cuda, params_dirname="model"):
    """使用模型测试"""
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = Inferencer(infer_func=image_classification_network, param_path=params_dirname, place=place)

    img = utils.load_rgb_image(CUR_PATH + '/data/image/infer_dog.png')

    # 预测
    results = inferencer.infer({'img': img})

    label_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print("infer results: %s" % label_list[numpy.argmax(results[0])])


if __name__ == '__main__':
    use_cuda = False
    # train
    train(use_cuda=use_cuda)
    # infer
    infer(use_cuda=use_cuda)
