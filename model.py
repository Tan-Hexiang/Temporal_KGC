from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import myDataset
from time2vec import T2Vlayer

class mymodel:
    def __init__(self,dataset,params) -> None:
        '''
        params.pretrained_embedding_path: TransE或者其他模型训练出的model的位置
        '''
        self.dataset = dataset
        self.params = params
        # self.ent_embs      = nn.Embedding(dataset.numEnt(), params.emb_dim).cuda()
        # self.rel_embs      = nn.Embedding(dataset.numRel(), params.emb_dim).cuda()
        # 导入TransE的embedding参数
        pretrained_model=torch.load(params.pretrained_embedding_path)
        self.ent_embs=nn.Embedding.from_pretrained(pretrained_model.state_dict()['ent_embs.weight']).cuda()
        self.rel_embs=nn.Embedding.from_pretrained(pretrained_model.state_dict()['rel_embs.weight']).cuda()
        # freeze the static embedding
        self.ent_embs.weight.requires_grad=False
        self.rel_embs.weight.requires_grad=False
        # t2v层
        self.start_t2v=nn.ModuleList([T2Vlayer(params.time_vec_dim) for i in range(params.granlarity_dim)])
        self.end_t2v=nn.ModuleList([T2Vlayer(params.time_vec_dim) for i in range(params.granlarity_dim)])
        # concatenate time ->params.time_vec_dim
        self.start_linear=nn.Linear(params.granlarity_dim*params.time_vec_dim,params.time_vec_dim)
        self.end_linear=nn.Linear(params.granlarity_dim*params.time_vec_dim,params.time_vec_dim)
        self.time_linear=nn.Linear(2*params.time_vec_dim,params.time_vec_dim)
        # concatenate all
        self.layer1=nn.Linear(params.time_vec_dim+3*self.ent_embs.embedding_dim,params.hidden_dim)
        self.layer2=nn.Linear(params.hidden_dim,params.hidden_dim)
        self.layer3=nn.Linear(params.hidden_dim,1)
        # activate_fu
        self.activate_fu=nn.ReLU()

    def forward(self, heads, rels, tails, start_time, end_time):
        # 获取TransE embedding
        h,r,t=self.getStaticEmbeddings(heads, rels, tails)
        start_gra=[]
        end_gra=[]
        # time2vec对每个粒度生成时间向量
        for i in range(self.params.granularity):
            start_gra.append(self.start_t2v[i](start_time[:,i]))
            end_gra.append(self.end_t2v[i](end_time[:,i]))
        # 连接不同粒度的时间
        start_time=torch.cat((start_gra[0],start_gra[1]),dim=1)
        end_time=torch.cat((end_gra[0],end_gra[1]),dim=1)
        for i in range(2,self.params.granularity):
            start_time=torch.cat((start_time,start_gra[i]),dim=1)
            end_time=torch.cat((end_time,end_gra[i]),dim=1)
        time=torch.cat((start_time,end_time),dim=1)
        # 连接时间向量与三元组
        time=self.time_linear(time)
        hidden_input=torch.cat((h,r,t,time),dim=1)
        # 三层MLP
        hidden=self.layer1(hidden_input)
        hidden=self.activate_fu(hidden)
        hidden=self.layer2(hidden)
        hidden=self.activate_fu(hidden)
        hidden=self.layer3(hidden)
        score=self.activate_fu(hidden)
        return score
    
    def getStaticEmbeddings(self, heads, rels, tails):
        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        return h,r,t

if __name__=='__main__':
    print('model.py')
    dic=[]
    dic.append(torch.tensor([1,2,3,4]).reshape(4,1))
    dic.append(torch.tensor([5,6,7,8]).reshape(4,1))
    dic.append(torch.tensor([9,1,2,3]).reshape(4,1))
    print(dic)
    time=torch.cat((dic[0],dic[1]),dim=1)
    print(time)
    time=torch.cat((time,dic[2]),dim=1)
    print(time)