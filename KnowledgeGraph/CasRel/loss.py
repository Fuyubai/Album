#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-07 16:07:16
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class UniteLoss(nn.Module):
    def __init__(self):
        super(UniteLoss, self).__init__()

    def forward(self, pred_subs, sub_labels, pred_rels, rel_labels, mask):
        sub_loss = F.binary_cross_entropy(pred_subs, sub_labels.float(), reduction='none')
        rel_labels = F.binary_cross_entropy(pred_rels, rel_labels.float())
        sub_loss = torch.sum(sub_loss * mask) / torch.sum(mask)
        loss = sub_loss + rel_labels
        return loss

if __name__ == '__main__':
    l = UniteLoss()
    pred_subs = torch.rand((128,128))
    sub_labels = torch.rand((128,128))
    pred_rels = torch.rand((128,9))
    rel_labels = torch.rand((128, 9))
    mask = torch.rand((128,128))
    a = l(pred_subs, sub_labels, pred_rels, rel_labels, mask)
