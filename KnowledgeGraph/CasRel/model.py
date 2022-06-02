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
            num_rels):
        super(Casrel, self).__init__()
        self.dim = dim
        self.num_rels = num_rels
        self.bert = BertModel.from_pretrained('/data/aleph_data/pretrained/TinyBERT_4L_zh/')
        self.sub_linear = nn.Linear(self.dim, 1)
        self.rel_linear = nn.Linear(self.dim, self.num_rels)

    def get_pred_subs(self, data):
        input_ids = data['input_ids'] 
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']

        output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        pred_subs = self.sub_linear(output[0])
        pred_subs = torch.sigmoid(pred_subs).squeeze(2)
        
        return output, pred_subs

    def get_pred_rels(self, output, sub_labels):
        sub_labels_ = sub_labels.unsqueeze(1).float()
        sub_embed = torch.matmul(sub_labels_, output[0]).squeeze(1)
        sub_embed = sub_embed / torch.sum(sub_labels_, dim=-1)
        output_with_subs = sub_embed + output[1]

        pred_rels = self.rel_linear(output_with_subs)
        pred_rels = torch.sigmoid(pred_rels)

        return pred_rels

    def forward(self, data, sub_labels):
        output, pred_subs = self.get_pred_subs(data)
        pred_rels = self.get_pred_rels(output, sub_labels)
        # input_ids = data['input_ids'] 
        # token_type_ids = data['token_type_ids']
        # attention_mask = data['attention_mask']

        # output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        # pred_subs = self.sub_linear(output[0])
        # pred_subs = torch.sigmoid(pred_subs).squeeze(2)
        
        # sub_labels_ = sub_labels.unsqueeze(1).float()
        # sub_embed = torch.matmul(sub_labels_, output[0]).squeeze(1)
        # sub_embed = sub_embed / torch.sum(sub_labels_, dim=-1)
        # output_with_subs = sub_embed + output[1]

        # pred_rels = self.rel_linear(output_with_subs)
        # pred_rels = torch.sigmoid(pred_rels)

        return pred_subs, pred_rels

if __name__ == '__main__':
    from dataloader import CasDataset
    dataset = CasDataset(
            'make_data/data/datas.json', 
            '/data/aleph_data/pretrained/TinyBERT_4L_zh/'
            )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for data in dataloader:
        break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Casrel(312, len(dataset.rel_mapping)).to(device)
    model(data[0], data[1])

    

