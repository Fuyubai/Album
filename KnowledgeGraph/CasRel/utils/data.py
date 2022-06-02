#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-05-31 20:06:02
'''
import json

def read_json(file):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            yield data

def dump_json(datas, file):
    with open(file, 'w') as f:
        for data in datas:
            data = json.dumps(data, ensure_ascii=False)
            f.write(data)
            f.write('\n')


