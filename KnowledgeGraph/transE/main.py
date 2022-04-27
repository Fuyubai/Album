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
        batch_size=1000
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
        dim=200,
        p_norm=1
        )

loss = MarginLoss(5.0)

trainer = Trainer(
        model,
        train_dataloader,
        test_dataloader,
        loss,
        epochs=100
        )

best_model_path = trainer.train()
trainer.load_model(best_model_path)
top10_head, mr_head = trainer.evaluate(0, False)
top10_tail, mr_tail = trainer.evaluate(1, False)

print('head')
print('top10: {:.4f}, mean rank: {}'.format(top10_head, mr_head))
print('tail')
print('top10: {:.4f}, mean rank: {}'.format(top10_tail, mr_tail))
print('head')
print('top10: {:.4f}, mean rank: {}'.format((top10_head+top10_tail)/2, (mr_head+mr_tail)/2))







