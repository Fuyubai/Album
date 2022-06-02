#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2021-08-13 12:00:44
'''


import torch
from torch.utils.data import DataLoader
import json
from transformers import AutoTokenizer

class CasDataset(torch.utils.data.Dataset):
    def __init__(self, 
            file, 
            tokenizer_root, 
            max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root)
        self.raw_datas = []
        self.encodings = []
        self.sub_labels = []
        self.rel_labels = []
        self.rels_mapping = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        datas = []
        if type(file) == str:
            datas = self.read_json(file)
        elif type(file) == list:
            datas = file
        
        for data in datas:
            self.raw_datas.append(data)
            text = self.clean_str(data['text']) 
            self.encodings.append(
                    self.tokenizer(
                        text,
                        max_length=max_length, 
                        truncation=True, 
                        padding='max_length',
                        return_tensors='pt'
                    )
                )

            tmp = torch.zeros((max_length, ), dtype=torch.int64)
            for entity in data['entities']:
                for i in range(entity['start'], entity['end']+1):
                    tmp[i] = 1
            self.sub_labels.append(tmp)
            self.rel_labels.append(data['rel'])

            # if len(self.raw_datas) > 1000:
            #     break

        self.rels_mapping = self.init_rels_mapping()

    def __getitem__(self, idx):
        datas = {k: torch.squeeze(v.to(self.device)) for k, v in self.encodings[idx].items()}
        sub_labels = self.sub_labels[idx].to(self.device)
        rel_labels = self.rel_labels[idx].to(self.device)
        return datas, sub_labels.to(self.device), rel_labels.to(self.device)

    def __len__(self):
        return len(self.rel_labels)

    @property
    def num_classes(self):
        tmp = [l.item() for l in self.labels]
        return len(set(tmp))
    
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
        s = s.replace('“', '')
        s = s.replace('”', '')
        s = s.replace('—', '')
        return s

    def init_rels_mapping(self):
        labels = sum(self.rel_labels, [])
        labels = list(set(labels))
        labels.sort()
        rels_mapping = {l: i for i, l in enumerate(labels)}
            
        new_rel_labels = []
        for rel in self.rel_labels:
            tmp = torch.zeros((len(rels_mapping), ), dtype=torch.int64)
            for r in rel:
                tmp[rels_mapping[r]] = 1
            new_rel_labels.append(tmp)

        self.rel_labels = new_rel_labels
        return rels_mapping

if __name__ == '__main__':
    dataset = CasDataset(
            'make_data/data/datas.json', 
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
            )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

