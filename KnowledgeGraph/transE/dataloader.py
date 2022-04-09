#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-07 11:09:01
'''
import os
import numpy as np
import random
from copy import deepcopy

class TrainDataloader():
    def __init__(
            self,
            data_dir,
            ent_file,
            rel_file,
            train_file,
            batch_size = 128,
            seed = 2021
            ):
        self.ent2id = {}
        self.rel2id = {}
        self.train_datas = []
       
        self.ent_count = None
        self.rel_count = None
        self.batch_size = batch_size
        random.seed(seed)
        random.shuffle(self.train_datas)

        with open(os.path.join(data_dir, ent_file), 'r') as f:
            line = f.readline().strip()
            self.ent_count = int(line)

            for line in f:
                line = line.strip().split('\t')
                self.ent2id[line[1]] = line[0]

        with open(os.path.join(data_dir, rel_file), 'r') as f:
            line = f.readline().strip()
            self.rel_count = int(line)

            for line in f:
                line = line.strip().split('\t')
                self.rel2id[line[1]] = line[0]

        with open(os.path.join(data_dir, train_file), 'r') as f:
            line = f.readline().strip()

            for line in f:
                line = line.strip().split()
                # head_ent, tail_ent, rel
                self.train_datas.append([line[0], line[1], line[2]])
        
    def __len__(self):
        return len(self.train_datas)

    def __iter__(self):
        for i in range((len(self.train_datas)//self.batch_size)+1):
            p_triples = self.train_datas[i*self.batch_size: (i+1)*self.batch_size]
            n_triples = []
            for triple in p_triples:
                n_triples.append(self.sample_corrupt(triple))
            yield np.array(p_triples+n_triples, dtype=np.int16)

    def sample_corrupt(self, triple):
        flag = random.choice([0, 1])
        tmp = triple[flag]
        while tmp == triple[flag]:
            tmp = random.randint(0, self.ent_count-1)
        corrupt_triple = deepcopy(triple)
        corrupt_triple[flag] = tmp
        return corrupt_triple

class TestDataloader():
    def __init__(
            self,
            data_dir,
            ent_file,
            rel_file,
            train_file,
            test_file,
            batch_size = 128,
            seed = 2021
            ):
        self.ent2id = {}
        self.rel2id = {}
        self.train_datas = []
        self.test_datas = []
        # for filtering corrupted test triples
        self.train_mapping = {}
        
        self.ent_count = None
        self.rel_count = None
        self.batch_size = batch_size
        random.seed(seed)
        random.shuffle(self.train_datas)

        with open(os.path.join(data_dir, ent_file), 'r') as f:
            line = f.readline().strip()
            self.ent_count = int(line)

            for line in f:
                line = line.strip().split('\t')
                self.ent2id[line[1]] = line[0]

        with open(os.path.join(data_dir, rel_file), 'r') as f:
            line = f.readline().strip()
            self.rel_count = int(line)

            for line in f:
                line = line.strip().split('\t')
                self.rel2id[line[1]] = line[0]

        with open(os.path.join(data_dir, train_file), 'r') as f:
            line = f.readline().strip()

            for line in f:
                line = line.strip().split()
                # head_ent, tail_ent, rel
                self.train_datas.append([line[0], line[1], line[2]])
        
        with open(os.path.join(data_dir, test_file), 'r') as f:
            line = f.readline().strip()

            for line in f:
                line = line.strip().split()
                # head_ent, tail_ent, rel
                self.test_datas.append([line[0], line[1], line[2]])
        
        self.init_train_mapping()

    def __len__(self):
        return len(self.test_datas)

    def batch_sample_corrupt(self, mode, need_filter=False):
        for data in self.test_datas:
            corrupted_triples = self.sample_corrupt(data, mode, need_filter)
            yield np.array(corrupted_triples, dtype=np.int64), np.array(data, dtype=np.int64)

    def sample_corrupt(self, triple, mode, need_filter=False):
        corrupted_triples = []
        for ent in self.ent2id:
            tmp = deepcopy(triple)
            tmp[mode] = ent
            if need_filter:
                if tmp[0] in self.train_mapping:
                    if tmp[1] in self.train_mapping[tmp[0]]:
                        if tmp[2] in self.train_mapping[tmp[0]][tmp[1]]:
                            continue
            corrupted_triples.append(tmp)
        return corrupted_triples

    def init_train_mapping(self):
        for data in self.train_datas:
            if data[0] not in self.train_mapping:
                self.train_mapping[data[0]] = {}
            if data[1] not in self.train_mapping[data[0]]:
                self.train_mapping[data[0]][data[1]] = set()
            self.train_mapping[data[0]][data[1]].add(data[2])       

if __name__ == '__main__':
    t = TrainDataloader(
            'data/FB15K237/',
            'entity2id.txt',
            'relation2id.txt',
            'train2id.txt',
            )
    
    t = TestDataloader(
            'data/FB15K237/',
            'entity2id.txt',
            'relation2id.txt',
            'train2id.txt',
            'test2id.txt'
            )
    
