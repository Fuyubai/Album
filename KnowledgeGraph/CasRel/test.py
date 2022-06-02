#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-05-31 14:25:25
'''

import time
from tqdm import tqdm
total_epoch = 10 
data_loader = range(100)
for epoch in range(total_epoch):
    with tqdm(total= len(data_loader)) as _tqdm: # 使用需要的参数对tqdm进行初始化
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch))# 设置前缀 一般为epoch的信息
        for data in data_loader:    
            time.sleep(0.01)
            _tqdm.set_postfix(loss='{:.6f}'.format(data)) # 设置你想要在本次循环内实时监视的变量  可以作为后缀打印出来
            _tqdm.update(1)
