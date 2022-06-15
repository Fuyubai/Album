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
from torch.utils.data.dataloader import default_collate

def filter_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

train_dataset = CasDataset(
            'data/CMED/test_triples.json',
            'data/CMED/rel2id.json',
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
            )
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=filter_none_collate)

dev_dataset = CasDataset(
            'data/CMED/dev_triples.json',
            'data/CMED/rel2id.json',
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/',
            is_test=True
            )
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True, collate_fn=filter_none_collate)

test_dataset = CasDataset(
            'data/CMED/dev_triples.json',
            'data/CMED/rel2id.json',
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/',
            is_test=True
            )
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=filter_none_collate)

model = Casrel(312, len(train_dataset.rel2id))

loss = UniteLoss()

trainer = Trainer(
        model,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        loss,
        epochs=20,
        save_dir='out/',
        tokenizer_root='/data/aleph_data/pretrained/TinyBERT_4L_zh/',
        rel_path='data/CMED/rel2id.json'
        )

trainer.train()
trainer.load_best_model()
metrics = trainer.evaluate(is_test=True)
print(metrics)

