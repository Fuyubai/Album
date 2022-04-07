#!/usr/bin/env python
#--coding:utf-8--
'''
    ~~~~~~~~~~~~~~~~~~
    author: Morning
    copyright: (c) 2021, Tungee
    date created: 2022-04-07 14:50:22
'''

import torch
from torch import nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(
            self,
            ent_count,
            rel_count,
            dim = 200,
            p_norm = 2,
            ):
        super(TransE, self).__init__()

        self.ent_count = ent_count
        self.rel_count = rel_count
        self.dim = dim
        self.p_norm = p_norm
        self.ent_embeddings = nn.Embedding(self.ent_count, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_count, self.dim)

        nn.init.uniform_(self.ent_embeddings.weight.data, -6/(self.dim)**0.5, 6/(self.dim)**0.5)
        nn.init.uniform_(self.rel_embeddings.weight.data, -6/(self.dim)**0.5, 6/(self.dim)**0.5)
        
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)

    def forward(self, inp):
        inp = torch.tensor(inp)
        inp = torch.transpose(inp, 0, 1)
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
        
        h = self.ent_embeddings(inp[0])
        t = self.ent_embeddings(inp[1])
        r = self.rel_embeddings(inp[2])

        score = h + r - t
        score = torch.norm(score, self.p_norm, -1)
        return score

if __name__ == '__main__':
    transe = TransE(10000, 200)
    import numpy as np
    tmp = np.random.randint(0, 100, [256, 3])
    


