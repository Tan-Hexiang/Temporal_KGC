from html import entities
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from scripts import shredFacts
'''
此类完成数据的读取
'''
class myDataset:
    def __init__(self,datapath,dataname,granularity_dim,valid_ratio,test_ratio) -> None:
        '''
        dataname= wiki 或 yoga
        '''
        super().__init__()
        # 读取时间五元组
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.name=dataname
        self.granularity_dim=granularity_dim  
        path=datapath+dataname+'/temporal'
        self.data=self.readtemporal(path,self.granularity_dim)
        # 全局变量，记录当前batch的指针位置
        self.start_batch=0
        # 读取实体和关系对照表
        self.entities,self.entity_ids=self.readentities(datapath+dataname+'/')
        self.relations,self.relation_ids=self.readrelations(datapath+dataname+'/')

        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])

    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data['train']):
            ret_facts = self.data['train'][self.start_batch : ]
            self.start_batch = 0
        else:
            ret_facts = self.data['train'][self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts

    def addNegSamples(self, bp_facts, neg_ratio):
        '''
        随机生成(s,r,?,t)和(?,r,o,t)的负样本
        input:
            bp_facts: shape(params.bsize, 元组长度),每行为一个（s,r,o,t)的多元组，其中t可以为多粒度
            neg_ratio: 每个正样本对应的负样本数量
        output:
            facts1: (?,r,o,t)的负样本,shape=( bsize*(1+neg_ratio), 元组长度 ), bsize组数据,每组一个正样本，neg_ratio个负样本
            facts2: (s,r,?,t)的负样本，同上
            输出为facts1和facts2按行堆叠的结果
        '''
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=0, high=self.numEnt()-1, size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=0, high=self.numEnt()-1, size=facts2.shape[0])
        # 随机从entity id list中取一个id
        for i in range(facts1.shape[0]):
            rand_nums1[i]=self.entity_ids[rand_nums1[i]]
            rand_nums2[i]=self.entity_ids[rand_nums2[i]]
        # 每组第一个为正样本
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i*pos_neg_group_size] = facts1[i*pos_neg_group_size,0]
            rand_nums2[i*pos_neg_group_size] = facts2[i*pos_neg_group_size,2]
        # 随机改变?位置的实体id
        facts1[:,0] = rand_nums1
        facts2[:,2] = rand_nums2
        return np.concatenate((facts1, facts2), axis=0)

    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegSamples(bp_facts, neg_ratio),self.granularity_dim)
        return batch

    def wasLastBatch(self):
        return (self.start_batch == 0)

    def readtemporal(self, filepath, granularity_dim) ->list:
        '''
        filepath: 数据完整路径
        output: 五元组[s,r,t,start_time,end_time]的list,
                其中start_time和end_time为granularity_dim维,缺失的粒度用0补充

        '''
        with open(filepath,'r') as f:
            data=f.readlines()

        result=[]
        for line in data:
            elements=line.strip().split('\t')

            s=int(elements[0])
            r=int(elements[1])
            t=int(elements[2])
            start_time=self.converttime(elements[3],granularity_dim) 
            end_time=self.converttime(elements[4],granularity_dim)
            triple=[s,r,t]
            result.append(np.concatenate((triple,start_time,end_time),axis=0))

        data_dic=self.split_valid_and_test(result)
        return data_dic

     

    def converttime(self,timestr,granularity_dim):
        '''
        timestr:data文件中的时间数据
        granularity_dim:输出结果的维度
        output: timelist
        '''
        timelist=[]
        # None值处理待考虑：暂时使用0值代替
        for i in range(granularity_dim):
            timelist.append(0)
        if timestr=='None':
            return timelist

        time=timestr.split('_')
        for i,value in enumerate(time):
            timelist[i]=int(value)
        return timelist
       

        
    def readrelations(self,filepath):
        relations={}
        relation_ids=[]
        path=filepath+'relations.dict'
        with open(path,'r') as f:
            data=f.readlines()
        for line in data:
            line=line.split('\t')
            rid=line[0]
            name=line[1]
            relations[rid]=name
            relation_ids.append(rid)       
        return relations, relation_ids

    def readentities(self,filepath):
        entities={}
        entity_ids=[]
        path=filepath+'entities.dict'
        with open(path,'r') as f:
            data=f.readlines()
        for line in data:
            line=line.split('\t')
            rid=line[0]
            name=line[1]
            entities[rid]=name
            entity_ids.append(rid)       
        return entities, entity_ids

    def numEnt(self):
    
        return len(self.entity_ids)

    def numRel(self):
    
        return len(self.relation_ids)

    def split_valid_and_test(self,data):
        '''
        split train, valid and test dataset with a fixed ratio
        '''
        result={}
        
        length=len(data)
        point1=int(self.test_ratio*length)
        point2=int((self.test_ratio+self.valid_ratio)*length)

        result['test']=data[:point1]
        result['valid']=data[point1:point2]
        result['train']=data[point2:]
        return result

