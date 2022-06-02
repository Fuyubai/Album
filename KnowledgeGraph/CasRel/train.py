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

train_dataset = CasDataset(
        'make_data/data/train.json', 
        '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
        )
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_dataset.dump_json([train_dataset.rels_mapping], 'rels_mapping.json')

dev_dataset = CasDataset(
        'make_data/data/dev.json', 
        '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
        )
dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True)

test_dataset = CasDataset(
        'make_data/data/test.json', 
        '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
        )
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = Casrel(312, len(train_dataset.rels_mapping))

loss = UniteLoss()

trainer = Trainer(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        loss,
        epochs=10,
        save_dir='out/',
        tokenizer_root='/data/aleph_data/pretrained/TinyBERT_4L_zh/',
        rels_mapping='rels_mapping.json'
        )

#trainer.train()
trainer.load_best_model()
# f1_subs, f1_rels = trainer.evaluate(is_test=True)
# print(f1_subs)
# print(f1_rels)
subs, rels = trainer.predict('帮我查下贫液槽的资料')
print(subs)
print(rels)


