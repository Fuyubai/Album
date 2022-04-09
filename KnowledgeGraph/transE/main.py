#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-08 20:05:59
'''

from dataloader import TrainDataloader, TestDataloader
from model import TransE
from loss import MarginLoss
from trainer import Trainer

train_dataloader = TrainDataloader(
        'data/FB15K237/',
        'entity2id.txt',
        'relation2id.txt',
        'train2id.txt',
        batch_size=1024
        )

test_dataloader = TestDataloader(
        'data/FB15K237/',
        'entity2id.txt',
        'relation2id.txt',
        'train2id.txt',
        'test2id.txt',
        )

model = TransE(
        ent_count=train_dataloader.ent_count,
        rel_count=train_dataloader.rel_count,
        dim=200
        )

loss = MarginLoss(2.0)

trainer = Trainer(
        model,
        train_dataloader,
        test_dataloader,
        loss,
        epochs=1000
        )

#trainer.train()    
trainer.load_model('out/model.ckpt')
trainer.evaluate(0, False)

