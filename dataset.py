import torch
from torch.utils import data
from copy import deepcopy

from graph import TemporalGraph


class Example(object):
    """Defines each triple in TKG"""
    def __init__(self, triple, granularity_dim, entity_vocab, relation_vocab, time_vocab, example_idx):
        self.head_idx = entity_vocab(triple[0])
        self.relation_idx = relation_vocab(triple[1])
        self.tail_idx = entity_vocab(triple[2])
        # time id
        self.start_time_id = time_vocab(triple[3])
        self.end_time_id = time_vocab(triple[4])
        # time list
        self.start_time_list=self.converttime(triple[3],granularity_dim)
        self.end_time_list=self.converttime(triple[4],granularity_dim)
        self.example_idx = example_idx

        self.graph = None

    def converttime(self,timestr,granularity_dim):
        '''
        timestr:data文件中的时间数据
        granularity_dim:输出结果的维度
        output: timelist
        '''
        timelist=[]
        # None值处理待考虑：暂时使用0值代替
        for i in range(granularity_dim):
            timelist.append(1e-5)
        if timestr=='None' or timestr=='none':
            return timelist
        time=timestr.split('_')
        
        for i,value in enumerate(time):
            if value == '##':
                timelist[i]=1e-5
            else:
                timelist[i]=float(value)
        return timelist

class TKGDataset(data.Dataset):
    """Temporal KG Dataset Class"""
    def __init__(self, example_list, kg, device):
        self.example_list = example_list
        self.kg = kg
        self.device = device

    def __iter__(self):
        return iter(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        return example

    def __len__(self):
        return len(self.example_list)

    def collate(self, batched_examples):
        batch_heads, batch_relations, batch_tails, batch_start_times_id, batch_end_times_id, batch_start_times_list,batch_end_times_list, batch_graph, batch_ex_indices = [], [], [], [], [], [], [], [], []
        for example in batched_examples:
            batch_heads.append(example.head_idx)
            batch_relations.append(example.relation_idx)
            batch_tails.append(example.tail_idx)

            batch_start_times_list.append(example.start_time_list)
            batch_end_times_list.append(example.end_time_list)
            batch_start_times_id.append(example.start_time_id)
            batch_end_times_id.append(example.end_time_id)

            batch_ex_indices.append(example.example_idx)
        # time tensor shape=([bsize,granularity_dim])
        return {
            "head": torch.tensor(batch_heads),
            "relation": torch.tensor(batch_relations),
            "tail": torch.tensor(batch_tails),

            "start_time_id": torch.tensor(batch_start_times_id),
            "end_time_id": torch.tensor(batch_end_times_id),
            "start_time_list": torch.tensor(batch_start_times_list),
            "end_time_list": torch.tensor(batch_end_times_list),
            
            "example_idx": torch.tensor(batch_ex_indices),
            "graph": deepcopy(self.kg.graph)
        }


def get_datasets(filenames, args):
    '''每个file输出对应的dataset'''
    KG = TemporalGraph(filenames[0], args)
    datasets = []

    for fname in filenames:
        triples = open(fname, 'r').read().lower().splitlines()
        triples = list(map(lambda x: x.split("\t"), triples))

        example_list = []
        for i, triple in enumerate(triples):
            example_list.append(Example(triple, args.granularity_dim, KG.entity_vocab, KG.relation_vocab, KG.time_vocab, i))

        datasets.append(TKGDataset(example_list, KG, args.device))

    return datasets
