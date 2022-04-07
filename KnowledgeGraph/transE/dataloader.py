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
        self.datas = []
        
        self.ent_count = None
        self.rel_count = None
        self.batch_size = batch_size
        random.seed(seed)
        random.shuffle(self.datas)

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
                self.datas.append([line[0], line[1], line[2]])

    def __len__(self):
        return len(self.datas)

    def __iter__(self):
        for i in range((len(self.datas)//self.batch_size)+1):
            p_triples = self.datas[i*self.batch_size: (i+1)*self.batch_size]
            n_triples = []
            for triple in p_triples:
                n_triples.append(self.sample_corrupt(triple))
            yield np.array(p_triples+n_triples, dtype=np.int16)

    def sample_corrupt(self, triple):
        flag = random.choice([0, 1])
        tmp = triple[flag]
        while tmp == triple[flag]:
            tmp = random.randint(0, self.ent_count)
        corrupt_triple = deepcopy(triple)
        corrupt_triple[flag] = tmp
        return corrupt_triple

if __name__ == '__main__':
    t = TrainDataloader(
            'data/FB15K237/',
            'entity2id.txt',
            'relation2id.txt',
            'train2id.txt',
            )
    
