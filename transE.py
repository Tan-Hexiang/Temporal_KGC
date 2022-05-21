# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pyexpat import model
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import myDataset


class TransE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(TransE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def getEmbeddings(self, heads, rels,tails, intervals = None):
   
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        
        return h,r,t
    
    def forward(self, heads, rels, tails, start_time=None, end_time=None):
        '''
        多余的输入参数是为了使用同一个trainer
        '''
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails)
        
        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim = 1)
        return scores

    def show(self):
        for name, param in self.named_parameters():
            print(name,param)

if __name__ == '__main__':
    dataset=myDataset('/data/tanhexiang/temporal_link_prediction/data/','yago11k',3,0.1,0.1)
    params=Params()
    model=TransE(dataset,params)
    
    for param_tensor in model.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())



