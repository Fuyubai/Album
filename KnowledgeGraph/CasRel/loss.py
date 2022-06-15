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
        self.scale = 0.02
        self.decay = 0.001

    def forward(self,
                sub_heads, sub_tails, 
                pred_sub_heads, pred_sub_tails,
                obj_heads, obj_tails, 
                pred_obj_heads, pred_obj_tails,
                mask):
        mask = mask.unsqueeze(2)
        sub_heads_loss = F.binary_cross_entropy(pred_sub_heads, sub_heads.unsqueeze(2).float(), reduction='none')
        sub_tails_loss = F.binary_cross_entropy(pred_sub_tails, sub_tails.unsqueeze(2).float(), reduction='none')
        obj_heads_loss = F.binary_cross_entropy(pred_obj_heads, obj_heads.float(), reduction='none')
        obj_tails_loss = F.binary_cross_entropy(pred_obj_tails, obj_tails.float(), reduction='none')

        sub_heads_loss = torch.sum(sub_heads_loss * mask) / torch.sum(mask)
        sub_tails_loss = torch.sum(sub_tails_loss * mask) / torch.sum(mask)
        obj_heads_loss = self.scale * torch.sum(obj_heads_loss * mask) / torch.sum(mask)
        obj_tails_loss = self.scale * torch.sum(obj_tails_loss * mask) / torch.sum(mask)

        loss = sub_heads_loss + sub_tails_loss + obj_heads_loss + obj_tails_loss
        self.scale = min(self.scale + self.decay, 1.0)
        return sub_heads_loss + sub_tails_loss, obj_heads_loss + obj_tails_loss, loss

if __name__ == '__main__':
    l = UniteLoss()
    sub_heads = torch.rand((32,512))
    sub_tails = torch.rand((32,512))
    pred_sub_heads = torch.rand((32,512,1))
    pred_sub_tails = torch.rand((32,512,1))
    obj_heads = torch.rand((32,512,44))
    obj_tails = torch.rand((32,512,44))
    pred_obj_heads = torch.rand((32,512,44))
    pred_obj_tails = torch.rand((32,512,44))
    mask = torch.rand((32,512))
    a = l(sub_heads, sub_tails, 
                pred_sub_heads, pred_sub_tails,
                obj_heads, obj_tails, 
                pred_obj_heads, pred_obj_tails,
                mask)
