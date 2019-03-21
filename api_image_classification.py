#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: api_image_classification.py
Desc: 图像分类 ml模型 API 封装
Demo: 
    cd /home/work/piglab/webservice/service/ && nohup python api_image_classification.py > log/api_image_classification.log &
    #图像分类
    http://www.yanjingang.com:8022/piglab/image/classification?img_file=/home/work/piglab/machinelearning/image/classification/data/image/infer_dog.png

    ps aux | grep api_image_classification.py |grep -v grep| cut -c 9-15 | xargs kill -9
Author: yanjingang(yanjingang@mail.com)
Date: 2018/12/28 23:08
"""

import sys
import os
import json
import tornado.ioloop
import tornado.web
import tornado.httpserver

# PATH
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.realpath(CUR_PATH + '/../../')
sys.path.append(BASE_PATH)
#print(CUR_PATH, BASE_PATH)
from machinelearning.lib import logger
import infer as classification_infer

class ApiImageClassification(tornado.web.RequestHandler):
    """API逻辑封装"""

    def get(self):
        """get请求处理"""
        try:
            result = self.execute()
        except:
            logger.error('execute fail ' + logger.get_trace(), ApiImageClassification)
            result = {'code': 1, 'msg': '查询失败'}
        logger.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + '][' + str(result['data']) + ']', ApiImageClassification)
        self.write(json.dumps(result))

    def post(self):
        """post请求处理"""
        try:
            result = self.execute()
        except:
            logger.error('execute fail ' + logger.get_trace(), ApiImageClassification)
            result = {'code': 1, 'msg': '查询失败'}
        logger.info('API RES[' + self.request.path + '][' + self.request.method + ']['
                      + str(result['code']) + '][' + str(result['msg']) + ']', ApiImageClassification)
        self.write(json.dumps(result))

    def execute(self):
        """执行业务逻辑"""
        logger.info('API REQUEST INFO[' + self.request.path + '][' + self.request.method + ']['
                      + self.request.remote_ip + '][' + str(self.request.arguments) + ']', ApiImageClassification)
        img_file = self.get_argument('img_file', '')
        if img_file == '':
            return {'code': 2, 'msg': 'img_file不能为空'}
        res = {}

        try:
            ret, msg, res = classification_infer.infer(img_file)
            if ret != 0:
                logger.error('execute fail [' + img_file + '] ' + msg, ApiImageClassification)
                return {'code': 4, 'msg': '查询失败'}
        except:
            logger.error('execute fail [' + img_file + '] ' + logger.get_trace(), ApiImageClassification)
            return {'code': 5, 'msg': '查询失败'}

        # 组织返回格式
        return {'code': 0, 'msg': 'success', 'data': res}


if __name__ == '__main__':
    """服务入口"""
    port = 8022

    # 路由
    app = tornado.web.Application(
        handlers=[
            (r'/piglab/image/classification', ApiImageClassification)
            ]
    )

    # 启动服务
    http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()

