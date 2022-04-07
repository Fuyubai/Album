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

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = torch.tensor([margin])

    def forward(self, score):
        tmp = score.shape[0] // 2
        p_score = score[:tmp]
        n_score = score[tmp:]
        return self.margin + torch.max(p_score - n_score, -self.margin).mean()

if __name__ == '__main__':
    l = MarginLoss(6.0)
    tmp = torch.randn((256, ))
    a = l(tmp)
