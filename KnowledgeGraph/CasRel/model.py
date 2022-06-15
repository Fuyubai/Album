#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-07 14:50:22
'''

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

class Casrel(nn.Module):
    def __init__(
            self, 
            dim, 
            rels_num):
        super(Casrel, self).__init__()
        self.dim = dim
        self.rels_num = rels_num
        self.bert = BertModel.from_pretrained('/data/aleph_data/pretrained/TinyBERT_4L_zh/')
        self.sub_head_linear = nn.Linear(self.dim, 1)
        self.sub_tail_linear = nn.Linear(self.dim, 1)
        self.obj_head_linear = nn.Linear(self.dim, self.rels_num)
        self.obj_tail_linear = nn.Linear(self.dim, self.rels_num)

    def get_pred_subs(self, encoding):
        input_ids = encoding['input_ids'] 
        token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']

        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        pred_sub_heads = self.sub_head_linear(output[0])
        pred_sub_tails = self.sub_tail_linear(output[0])

        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        
        return output, pred_sub_heads, pred_sub_tails

    def get_pred_objs(self, output, pred_sub_heads, pred_sub_tails):
        pred_sub_heads_ = pred_sub_heads.unsqueeze(1).float()
        pred_sub_tails_ = pred_sub_tails.unsqueeze(1).float()

        pred_sub = pred_sub_heads_ + pred_sub_tails_
        sub_embed = torch.matmul(pred_sub, output[0])
        sub_embed = sub_embed / torch.sum(pred_sub, dim=-1).unsqueeze(2)
        output_with_subs = sub_embed + output[0]

        pred_obj_heads = self.obj_head_linear(output_with_subs)
        pred_obj_tails = self.obj_tail_linear(output_with_subs)
        
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
    
        return pred_obj_heads, pred_obj_tails

    def forward(self, encoding, sub_head, sub_tail):
        output, pred_sub_heads, pred_sub_tails = self.get_pred_subs(encoding)
        pred_obj_heads, pred_obj_tails = self.get_pred_objs(output, sub_head, sub_tail)
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails

if __name__ == '__main__':
    from dataloader import CasDataset
    dataset = CasDataset(
            'data/CMED/dev_triples.json',
            'data/CMED/rel2id.json',
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
            )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in dataloader:
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Casrel(312, len(dataset.rel2id)).to(device)
    model(data[0], data[3], data[4])

    

