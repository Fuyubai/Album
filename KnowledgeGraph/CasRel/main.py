#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-08 20:05:59
'''

from dataloader import CasDataset
from model import Casrel
from loss import UniteLoss
from trainer import Trainer
from torch.utils.data import DataLoader
from nebula.client import NebulaClient
from utils.data import read_json
from tqdm import tqdm
import time

if __name__ == '__main__':

    model = Casrel(312, 9)
    trainer = Trainer(
            model,
            save_dir='out/',
            tokenizer_root='/data/aleph_data/pretrained/TinyBERT_4L_zh/',
            rels_mapping='rels_mapping.json'
            )
    trainer.load_best_model()

    nclient = NebulaClient(
            '127.0.0.1',
            9669,
            'root',
            'a',
            'test3'
            )

    alias_dict = list(read_json('data/alias_dict.json'))[0]

    datas = read_json('data/test.json') 
    c = 0
    s = time.time()
    for data in tqdm(datas):
        import ipdb;ipdb.set_trace()
        subs, rels = trainer.predict(data['text'])
        results = []
        for sub in subs:
            for rel in rels:
                ents = alias_dict[sub]
                for ent in ents:
                    nsql = nclient.get_single_ent_and_single_rel(ent, rel)
                    result = nclient.execute(nsql)
                    for r in result:
                        t_ent = r.get('row')[0]
                        if t_ent and rel == 'relate_tech_params':
                            results.append('参数')
                            continue
                        for k, v in t_ent.items():
                            if k.endswith('.name'):
                                results.append(v)
        if data['label'] in results:
            c += 1
        else:
            print(data['label'])
            print(data['text'])
            print(results)

    e = time.time()    
    print(e-s)
        

    print(c/len(data))
    nclient.close()

