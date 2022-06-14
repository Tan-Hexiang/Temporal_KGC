import dgl
import torch
from util import Vocab
'''对dgl.graph进行的一层封装，增加了vocab和从文件创建的处理过程'''

class TemporalGraph:
    """Temporal Graph Container Class"""
    def __init__(self, train_path, args):
        self.device = args.device

        with open(train_path, 'r') as f:
            lines = f.read().lower().splitlines()
            lines = map(lambda x: x.split("\t"), lines)

            head_list, relation_list, tail_list, start_time_id, end_time_id= tuple(zip(*lines)) #zip(*)解压为列表，lines中的是一行行的，返回的结果是一列列的
            self.entity_vocab = Vocab()
            self.relation_vocab = Vocab()
            self.time_vocab = Vocab()
            self.entity_vocab.update(head_list + tail_list)
            self.relation_vocab.update(relation_list)
            self.time_vocab.update(start_time_id)
            self.time_vocab.update(end_time_id)
            self.entity_vocab.build()
            self.relation_vocab.build()
            self.time_vocab.build(sort_key="time")
            # time list 
            start_time_list=list(map(lambda x: self.converttime(x,args.granularity_dim),start_time_id))
            end_time_list=list(map(lambda x: self.converttime(x,args.granularity_dim),end_time_id))
            # 将list映射到vocab
            head_list = list(map(lambda x: self.entity_vocab(x), head_list))
            relation_list = list(map(lambda x: self.relation_vocab(x), relation_list))
            tail_list = list(map(lambda x: self.entity_vocab(x), tail_list))
            start_time_id = list(map(lambda x: self.time_vocab(x), start_time_id))
            end_time_id=list(map(lambda x:self.time_vocab(x),end_time_id))

            print("time vacob:\n")
            print(self.time_vocab)

        self.graph = dgl.DGLGraph(multigraph=True)
        self.graph.add_nodes(len(self.entity_vocab))
        self.graph.add_edges(head_list, tail_list)
        self.graph.ndata['node_idx'] = torch.arange(self.graph.number_of_nodes())
        self.graph.edata['relation_type'] = torch.tensor(relation_list)
        # vocab time
        self.graph.edata['start_time_id'] = torch.tensor(start_time_id)
        self.graph.edata['end_time_id'] = torch.tensor(end_time_id)
        # list time
        self.graph.edata['start_time_list'] = torch.tensor(start_time_list)
        self.graph.edata['end_time_list'] = torch.tensor(end_time_list)
        print("Graph prepared.")
    
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
        if timestr=='None' or timestr=='none':
            return timelist
        time=timestr.split('_')
        
        for i,value in enumerate(time):
            if value == '##':
                timelist[i]=0
            else:
                timelist[i]=int(value)
        return timelist
