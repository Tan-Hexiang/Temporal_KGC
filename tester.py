import numpy as np
import torch 
from measure import Measure
from scripts import shredFacts
'''
Hit@ N MRR计算:https://blog.csdn.net/qq_36158230/article/details/120254381
'''
class Tester:
    def __init__(self,modelpath,dataset,test_or_valid='valid') -> None:
        self.model=torch.load(modelpath)
        self.model.eval()
        self.dataset=dataset
        self.measure=Measure()
        self.test_or_valid=test_or_valid
     

    def getRank(self,sim_scores):
        '''
        sim_scores: the similarity scores for every candidate entities
        output:     the rank of the first score
        '''
        print("\nget score:",sim_scores)
        print("\nreturn rank:",(sim_scores > sim_scores[0]).sum() + 1)
        return (sim_scores > sim_scores[0]).sum() + 1
        
    # def replaceAndShred(self, fact, raw_or_fil, head_or_tail):
        
    #     head, rel, tail, start_time, end_time = fact
    #     if head_or_tail == "head":
    #         ret_facts = [(i, rel, tail, start_time, end_time) for i in range(self.dataset.numEnt())]
    #     if head_or_tail == "tail":
    #         ret_facts = [(head, rel, i, start_time, end_time) for i in range(self.dataset.numEnt())]
    #     # 第一行为真实五元组
    #     if raw_or_fil == "raw":
    #         ret_facts = [tuple(fact)] + ret_facts
    #     # elif raw_or_fil == "fil":
    #     #     ret_facts = [tuple(fact)] + list(set(ret_facts) - self.dataset.all_facts_as_tuples)
    #     # 注意这里除了第一行为真实的五元组之外，下面还有一个真实五元组。但是由于getRank中计算时使用的>而不是>=,所以对rank没有影响
    #     return shredFacts(np.array(ret_facts),self.dataset.granularity_dim)

    def replaceAndShred2(self, fact, raw_or_fil, head_or_tail):
        # 不用所有的实体，取部分实体做候选集
        facts1 = np.repeat(np.copy(fact).reshape((1,len(fact))), self.dataset.numEnt()+1, axis=0)
        if head_or_tail == "head":
            for i in range(1,self.dataset.numEnt()+1):
                facts1[i,0]=self.dataset.entity_ids[i-1]
        if head_or_tail == "tail":
            for i in range(1,self.dataset.numEnt()+1):
                facts1[i,2]=self.dataset.entity_ids[i-1]
        # 第一行为真实五元组
        # if raw_or_fil == "raw":
        #     facts1 = [tuple(fact)] + facts1
        # elif raw_or_fil == "fil":
        #     facts1 = [tuple(fact)] + list(set(facts1) - self.dataset.all_facts_as_tuples)
        # 注意这里除了第一行为真实的五元组之外，下面还有一个真实五元组。但是由于getRank中计算时使用的>而不是>=,所以对rank没有影响
        # print("\nget fact:",fact)
        # print("\nreturn ret_facts:",facts1[0],facts1[1],facts1[3])
        return shredFacts(np.array(facts1),self.dataset.granularity_dim)

    def test(self):
        for i, fact in enumerate(self.dataset.data[self.test_or_valid]):
            settings = ["raw"]
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:     
                    heads, rels, tails, start_time, end_time = self.replaceAndShred2(fact, raw_or_fil, head_or_tail)
                    sim_scores = self.model(heads, rels, tails, start_time, end_time).cpu().data.numpy()
                    rank = self.getRank(sim_scores)
                    self.measure.update(rank, raw_or_fil)
                    
        
        self.measure.print_()
        print("~~~~~~~~~~~~~")
        self.measure.normalize(len(self.dataset.data[self.test_or_valid]))
        self.measure.print_()
        
        return self.measure.mrr["raw"]

if __name__=='__main__':
    x=[1,2,3,4]

    facts=np.repeat(np.copy(x).reshape((1,len(x))),2,axis=0)
    print(facts)