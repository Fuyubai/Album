#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-08 17:49:00
'''

import os
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            test_dataloader,
            loss,
            save_dir='out/',
            epochs=10,
            ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.loss = loss
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss.to(self.device)
        self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=0.01,
                momentum=0.9)

    def train_one_epoch(self):
        for i, data in enumerate(self.train_dataloader):
            data = torch.tensor(data, dtype=torch.int64).to(self.device)
            score = self.model(data)
            l = self.loss(score)
            l.backward()
            self.optimizer.step()
        return l

    def train(self):
        self.model.train()
        min_loss = 10000000
        train_range = tqdm(range(self.epochs))
        for epoch in train_range:
            loss = self.train_one_epoch()
            train_range.set_description("Epoch %d | loss: %f" % (epoch, loss))
            if loss < min_loss:
                min_loss = loss
                name = str(epoch) + '_' + str(loss.item()) + '.ckpt'
                path = os.path.join(self.save_dir, name)
                self.save_model(path)
        return path
    
    def evaluate(self, mode, need_filter=False):
        self.model.eval()
        top10 = 0
        mr = 0
        with torch.no_grad():
            for corrupted_triples, triple in tqdm(self.test_dataloader.batch_sample_corrupt(mode, need_filter)):
                data = torch.tensor(corrupted_triples, dtype=torch.int64).to(self.device)
                output = self.model(data)
                
                res = torch.cat((data[:, mode].reshape(-1, 1), output.reshape(-1, 1)), dim=-1)
                res = res[res[:, 1].sort()[1]]
                index = (res[:, 0] == triple[mode]).nonzero(as_tuple=True)[0].item()
                # res = list(zip(data_, output_))
                # res.sort(key=lambda x: x[1])
                # res = [r[0] for r in res]
                # index = res.index(triple[mode]) + 1
                if index <= 10:
                    top10 += 1
                mr += index
        
            top10 = top10 / len(self.test_dataloader)
            mr = int(mr / len(self.test_dataloader))
        
        return top10, mr

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
      


