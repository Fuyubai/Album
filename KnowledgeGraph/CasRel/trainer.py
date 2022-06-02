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
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from dataloader import CasDataset

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader=None,
            dev_dataloader=None,
            test_dataloader=None,
            loss=None,
            save_dir=None,
            epochs=10,
            tokenizer_root=None,
            rels_mapping=None,
            max_length=128):
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.loss = loss
        self.save_dir = save_dir
        self.max_length = max_length
        if tokenizer_root:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_root)
        if rels_mapping:
            self.rels_mapping = list(CasDataset.read_json(rels_mapping))[0]
            self.rels_mapping = {v: k for k, v in self.rels_mapping.items()}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=3e-4
            )

    def train_one_epoch(self, epoch):
        train_log = tqdm(self.train_dataloader)
        train_log.set_description('Epoch: {}/{}'.format(epoch+1, self.epochs))
        for data, sub_labels, rel_labels in train_log:
            pred_subs, pred_rels = self.model(data, sub_labels)
            l = self.loss(pred_subs, sub_labels, pred_rels, rel_labels, data['attention_mask'])
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_log.set_postfix(loss='{:4f}'.format(l.item()))
        return l

    def train(self):
        max_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            loss = self.train_one_epoch(epoch)
                        
            f1_subs, f1_rels = self.evaluate()
            print('val f1_subs: {:4f} val f1_rels: {:4f}'.format(f1_subs, f1_rels))

            f1 = f1_subs + f1_rels
            if f1 > max_f1:
                max_f1 = f1
                checkpoint = str(epoch) + '_' + str(round(f1_subs, 4)) + '_' + str(round(f1_rels, 4)) + '.ckpt'
                self.save_model(checkpoint)
    
    def evaluate(self, is_test=False):
        self.model.eval()
        f1_subs = 0
        f1_rels = 0
        if not is_test:
            dataloader = self.dev_dataloader
        else:
            dataloader = self.test_dataloader

        with torch.no_grad():
            for i, (data, sub_labels, rel_labels) in enumerate(dataloader):
                output, pred_subs = self.model.get_pred_subs(data)
                pred_subs = torch.where(pred_subs>=0.5, 1, 0) * data['attention_mask']
                pred_rels = self.model.get_pred_rels(output, pred_subs)
                pred_rels = torch.where(pred_rels>=0.5, 1, 0)
                f1_subs += f1_score(sub_labels.reshape(-1).cpu(), pred_subs.reshape(-1).cpu())
                f1_rels += f1_score(rel_labels.reshape(-1).cpu(), pred_rels.reshape(-1).cpu())
        
        f1_subs = f1_subs / (i+1)
        f1_rels = f1_rels / (i+1)
        return f1_subs, f1_rels

    def predict(self, query):
        def clean_str(s):
            s = s.replace('“', '')
            s = s.replace('”', '')
            s = s.replace('—', '')
            return s

        query_ = clean_str(query)
        encoding = self.tokenizer(
                    query_,
                    max_length=self.max_length, 
                    truncation=True, 
                    padding='max_length',
                    return_tensors='pt'
                )
        data = {k: v.to(self.device) for k, v in encoding.items()}
        
        output, pred_subs = self.model.get_pred_subs(data)
        pred_subs = torch.where(pred_subs>=0.5, 1, 0) * data['attention_mask']
        pred_rels = self.model.get_pred_rels(output, pred_subs)
        pred_rels = torch.where(pred_rels>=0.5, 1, 0)

        pred_subs = pred_subs.tolist()[0]
        pred_rels = pred_rels.tolist()[0]
        
        subs = []
        rels = []
        s = None
        e = None
        for i, tag in enumerate(pred_subs):
            if tag == 1:
                if s is None:
                    s = i
                else:
                    e = i
            if tag == 0:
                if s is not None and e is not None:
                    subs.append((s, e))
                    s = None
                    e = None
        subs = [query[s:e+1] for s, e in subs] 
        
        for i, tag in enumerate(pred_rels):
            if tag == 1:
                rels.append(self.rels_mapping[i])

        return subs, rels


    def save_model(self, checkpoint):
        path = os.path.join(self.save_dir, checkpoint)
        torch.save(self.model.state_dict(), path)
    
    def load_best_model(self, path=None):
        if path:
            self.model.load_state_dict(torch.load(path))
            return 

        checkpoints = os.listdir(self.save_dir)
        best_metric = 0
        best_checkpoint = None
        for checkpoint in checkpoints:
            _, f1_subs, f1_rels = checkpoint.split('.ckpt')[0].split('_')
            metric = float(f1_subs) + float(f1_rels)
            if metric > best_metric:
                best_metric = metric
                best_checkpoint = checkpoint

        print(best_checkpoint)
        path = os.path.join(self.save_dir, best_checkpoint)
        self.model.load_state_dict(torch.load(path))


