#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-08 17:49:00
'''

import os
import json
import numpy as np
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
            rel_path=None,
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
        if rel_path:
            self.rel2id = json.load(open(rel_path, 'r'))[0]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=3e-4
            )

    def train_one_epoch(self, epoch):
        total_loss = 0
        train_log = tqdm(self.train_dataloader)
        train_log.set_description('Train Epoch: {}/{}'.format(epoch+1, self.epochs))
        for i, data in enumerate(train_log):
            if data is None:
                continue
            encoding, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails = data
            pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = self.model(encoding, sub_head, sub_tail)

            s_l, o_l, l = self.loss(
                    sub_heads, sub_tails, 
                    pred_sub_heads, pred_sub_tails,
                    obj_heads, obj_tails, 
                    pred_obj_heads, pred_obj_tails,
                    encoding['attention_mask'])
            
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_log.set_postfix(loss='{:4f}'.format(l.item()))
            total_loss += l.item()

        total_loss = total_loss / (i + 1)
        return total_loss

    def train(self):
        max_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            loss = self.train_one_epoch(epoch)
                       
            _, _, _, _, _, f1_score = self.evaluate()
            print('val f1_score: {:4f}'.format(f1_score))

            if f1_score > max_f1:
                max_f1 = f1_score
                checkpoint = str(epoch) + '_' + str(round(f1_score, 4)) + '.ckpt'
                self.save_model(checkpoint)
    
    def evaluate(self, is_test=False):
        self.model.eval()
        f1_subs = 0
        f1_rels = 0
        if not is_test:
            dataloader = self.dev_dataloader
        else:
            dataloader = self.test_dataloader

        correct_num, gold_num, pred_num = 0, 0, 0
        with torch.no_grad():
            for data in tqdm(dataloader):
                if data is None:
                    continue
                encoding, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, sr2o_map = data
                output, pred_sub_heads, pred_sub_tails = self.model.get_pred_subs(encoding)
                pred_sub_heads = torch.where((pred_sub_heads.squeeze(2) * encoding['attention_mask'])[0] >= 0.5)[0]
                pred_sub_tails = torch.where((pred_sub_tails.squeeze(2) * encoding['attention_mask'])[0] >= 0.5)[0]
                
                subs = []
                for pred_sub_head in pred_sub_heads:
                    pred_sub_tail = pred_sub_tails[pred_sub_tails >= pred_sub_head]
                    if len(pred_sub_tail) > 0:
                        pred_sub_tail = pred_sub_tail[0]
                        subs.append((pred_sub_head.item(), pred_sub_tail.item()))

                pred_sr2o_map = {}
                if subs:
                    for sub_head_idx, sub_tail_idx in subs:
                        sub_head = np.zeros(sub_head.shape) 
                        sub_tail = np.zeros(sub_tail.shape)
                        sub_head[0][sub_head_idx] = 1
                        sub_tail[0][sub_tail_idx] = 1
                        sub_head = torch.tensor(sub_head, dtype=torch.int64).to(self.device)
                        sub_tail = torch.tensor(sub_tail, dtype=torch.int64).to(self.device)
                        
                        pred_obj_heads, pred_obj_tails = self.model.get_pred_objs(output, sub_head, sub_tail)
                        for i in range(pred_obj_heads[0].shape[-1]):
                            pred_obj_heads_ = pred_obj_heads[:, :, i]
                            pred_obj_tails_ = pred_obj_tails[:, :, i]
                            pred_obj_heads_ = torch.where((pred_obj_heads_ * encoding['attention_mask'])[0] >= 0.5)[0]
                            pred_obj_tails_ = torch.where((pred_obj_tails_ * encoding['attention_mask'])[0] >= 0.5)[0]
                            
                            for pred_obj_head in pred_obj_heads_:
                                pred_obj_tail = pred_obj_tails_[pred_obj_tails_ >= pred_obj_head]
                                if len(pred_obj_tail) > 0:
                                    pred_obj_tail = pred_obj_tail[0]
                                    if (sub_head_idx, sub_tail_idx) not in pred_sr2o_map:
                                        pred_sr2o_map[(sub_head_idx, sub_tail_idx)] = [[pred_obj_head.item(), pred_obj_tail.item(), i]]
                                    else:
                                        pred_sr2o_map[(sub_head_idx, sub_tail_idx)].append([pred_obj_head.item(), pred_obj_tail.item(), i])
                        
                gold_triples_set = set()
                pred_triples_set = set()
                for (s_h, s_t), v in sr2o_map.items():
                    for (o_h, o_t, r) in v:
                        tmp = (s_h, s_t, o_h.item(), o_t.item(), r.item())
                        gold_triples_set.add(tmp)
                for (s_h, s_t), v in pred_sr2o_map.items():
                    for (o_h, o_t, r) in v:
                        tmp = (s_h, s_t, o_h, o_t, r)
                        pred_triples_set.add(tmp)

                correct_num += len(gold_triples_set & pred_triples_set)
                gold_num += len(gold_triples_set)
                pred_num += len(pred_triples_set)

            precision = correct_num / (pred_num + 1e-10)
            recall = correct_num / (gold_num + 1e-10)
            f1_score = 2 * precision * recall / (precision + recall + 1e-10)
            return correct_num, gold_num, pred_num, precision, recall, f1_score

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
            _, f1_score = checkpoint.split('.ckpt')[0].split('_')
            metric = float(f1_score)
            if metric > best_metric:
                best_metric = metric
                best_checkpoint = checkpoint

        print(best_checkpoint)
        path = os.path.join(self.save_dir, best_checkpoint)
        self.model.load_state_dict(torch.load(path))


