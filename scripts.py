# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from tracemalloc import start
import torch

def shredFacts(facts,granularity_dim): #takes a batch of facts and shreds it into its columns
        
    heads      = torch.tensor(facts[:,0]).long().cuda()
    rels       = torch.tensor(facts[:,1]).long().cuda()
    tails      = torch.tensor(facts[:,2]).long().cuda()
    start_time = torch.tensor(facts[:,3:3+granularity_dim]).float().cuda()
    end_time = torch.tensor(facts[:,3+granularity_dim:]).float().cuda()
    return heads, rels, tails, start_time, end_time

# '''test'''
# facts=[1,2,3,4,5,6,7]

# print(facts[3:3+2])
# print(facts[3+2:])
