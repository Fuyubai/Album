#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2021-08-13 12:00:44
'''

import json
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class CasDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_path,
            rel_path,
            tokenizer_root,
            max_length=512,
            is_test=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root)
        self.raw_datas = json.load(open(data_path, 'r'))
        self.rel2id = json.load(open(rel_path, 'r'))[1]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encodings = []
        self.sub_heads = []
        self.sub_tails = []
        self.sr2o_maps = []
        self.sub_head = []
        self.sub_tail = []
        self.obj_heads = []
        self.obj_tails = []
        self.max_length = max_length
        self.is_test= is_test

        for data in self.raw_datas:
            text = ''.join(self.clean_str(data['text']))
            tokens = self.tokenizer.tokenize(text)
            self.encodings.append(
                    self.tokenizer(
                        text,
                        max_length=self.max_length, 
                        truncation=True, 
                        padding='max_length',
                        return_tensors='pt'
                    )
                )
            sr2o_map = {}

            for triple in data['triple_list']:
                triple = [self.clean_str(triple[0]), self.clean_str(triple[1]), self.clean_str(triple[2])]
                triple[0] = self.tokenizer.tokenize(triple[0])
                triple[2] = self.tokenizer.tokenize(triple[2])
                sub_head_idx = self.find_head_idx(tokens, triple[0])
                obj_head_idx = self.find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in sr2o_map:
                        sr2o_map[sub] = []
                    sr2o_map[sub].append((obj_head_idx, obj_head_idx + len(triple[2]) -1, self.rel2id[triple[1]]))

            self.sr2o_maps.append(sr2o_map)
                 
    def __getitem__(self, idx):
        datas = {k: torch.squeeze(v.to(self.device)) for k, v in self.encodings[idx].items()}
        sr2o_map = self.sr2o_maps[idx]

        # NER sqeuence label
        sub_heads = np.zeros(self.max_length)
        sub_tails = np.zeros(self.max_length)
        for (sub_head_idx, sub_tail_idx) in sr2o_map:
            sub_heads[sub_head_idx] = 1
            sub_tails[sub_tail_idx] = 1
        sub_heads = torch.tensor(sub_heads, dtype=torch.int64)
        sub_tails = torch.tensor(sub_tails, dtype=torch.int64)
        
        # when training, not all sub will attend to train, it choices a sub randomly,
        # then uses rels and objs of this sub to coustruct a matrix 
        # about [len_squence, len_label]
        sub_head = np.zeros(self.max_length)
        sub_tail = np.zeros(self.max_length)
        obj_heads = np.zeros((self.max_length, len(self.rel2id)))
        obj_tails = np.zeros((self.max_length, len(self.rel2id)))
        if sr2o_map:
            sr2o_map_key = random.choice(list(sr2o_map.keys()))
            sub_head[sr2o_map_key[0]] = 1
            sub_tail[sr2o_map_key[1]] = 1
            for obj_head, obj_tail, rel in sr2o_map[sr2o_map_key]:
                obj_heads[obj_head][rel] = 1
                obj_tails[obj_tail][rel] = 1
        else:
            return None

        sub_head = torch.tensor(sub_head, dtype=torch.int64)
        sub_tail = torch.tensor(sub_tail, dtype=torch.int64)

        obj_heads = torch.tensor(obj_heads, dtype=torch.int64)
        obj_tails = torch.tensor(obj_tails, dtype=torch.int64)

        if not self.is_test:
            return datas, sub_heads.to(self.device), sub_tails.to(self.device), sub_head.to(self.device), sub_tail.to(self.device), obj_heads.to(self.device), obj_tails.to(self.device)
        else:
            return datas, sub_heads.to(self.device), sub_tails.to(self.device), sub_head.to(self.device), sub_tail.to(self.device), obj_heads.to(self.device), obj_tails.to(self.device), sr2o_map
    
    def __len__(self):
        return len(self.raw_datas)
    
    @staticmethod
    def read_json(file):
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                yield data
    
    @staticmethod
    def dump_json(datas, file):
        with open(file, 'w') as f:
            for data in datas:
                data = json.dumps(data, ensure_ascii=False)
                f.write(data)
                f.write('\n')

    @staticmethod
    def clean_str(s):
        s = ''.join(s.split())
        return s

    def find_head_idx(self, source, target):
        for i in range(len(source)):
            if source[i: i+len(target)] == target:
                return i+1
        return -1


if __name__ == '__main__':
    dataset = CasDataset(
            'data/CMED/dev_triples.json',
            'data/CMED/rel2id.json',
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
            )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    dataset[0]


