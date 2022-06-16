import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time2vec import T2Vlayer


class TGAP(nn.Module):
    def __init__(self, args):
        super(TGAP, self).__init__()
        self.args = args
        self.num_out_heads = args.num_out_heads
        self.num_in_heads = args.num_in_heads
        self.out_head_dim = args.node_dim // self.num_out_heads
        self.in_head_dim = args.node_dim // self.num_in_heads

        # Entity, Relation, Timestamp Embeddings
        self.node_embed = NodeEmbedding(args)
        self.edge_embed = nn.Embedding(len(args.relation_vocab), args.node_dim, padding_idx=0)
        # self.tau_embed = nn.Embedding(len(args.time_vocab), args.node_dim, padding_idx=0)

        # time2vec layer: 每个粒度的向量维度不同
        # self.start_t2v=nn.ModuleList([T2Vlayer(args.node_dim) for i in range(args.granularity_dim)])
        # self.end_t2v=nn.ModuleList([T2Vlayer(args.node_dim) for i in range(args.granularity_dim)])
        self.start_t2v=nn.ModuleList([])
        self.end_t2v=nn.ModuleList([])
        for i in range(args.granularity_dim):
            self.start_t2v.append(T2Vlayer(args.time_dim[i]))
            self.end_t2v.append(T2Vlayer(args.time_dim[i]))
    
        # linear layers of time
        self.time_linear=nn.Linear(2*args.all_time_dim, args.node_dim)
        self.time_score=nn.Sequential(
            nn.Linear(args.node_dim*2,args.node_dim*4),
            nn.ReLU(),
            nn.Linear(args.node_dim*4,args.node_dim*2),
            nn.ReLU(),
            nn.Linear(args.node_dim*2,args.node_dim)
        )
        
        # Linear Layers
        self.W_c = nn.Linear(args.node_dim * 2, args.node_dim)
        self.W_n = nn.Linear(args.node_dim * 2, args.node_dim)
        self.W_h = nn.Linear(args.node_dim, args.node_dim)

        # Attention Heads for Attention Flow
        self.attn_i_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.attn_j_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.inattn_i_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.inattn_j_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))

        # Attention Heads for PGNN
        self.PGNN_attn_i_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))
        self.PGNN_attn_j_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))

        # Attention Heads for SGNN
        self.SGNN_attn_i_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))
        self.SGNN_attn_j_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))

        nn.init.xavier_uniform_(self.attn_i_outgoing)
        nn.init.xavier_uniform_(self.attn_j_outgoing)
        nn.init.xavier_uniform_(self.inattn_i_outgoing)
        nn.init.xavier_uniform_(self.inattn_j_outgoing)
        nn.init.xavier_uniform_(self.PGNN_attn_i_incoming)
        nn.init.xavier_uniform_(self.PGNN_attn_j_incoming)
        nn.init.xavier_uniform_(self.SGNN_attn_i_incoming)
        nn.init.xavier_uniform_(self.SGNN_attn_j_incoming)

        # Timestamp Sign Parameters
        # time order W, for example: s_s_ used in s_s_time 
        self.inattn_s_s_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_s_s_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_s_s_future_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_e_e_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_e_e_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_e_e_future_lin = nn.Linear(args.node_dim, args.node_dim)

        self.attn_s_s_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_s_s_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_s_s_future_lin =  nn.Linear(args.node_dim, args.node_dim)
        self.attn_e_e_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_e_e_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_e_e_future_lin =  nn.Linear(args.node_dim, args.node_dim)


    def forward(self, batch):
        batch_size = batch["head"].size(0)

        # Prepare graph
        graph = batch['graph'].local_var().to(self.args.device)  #一个副本
        if self.training:   # 对于整图数据，需要在训练时将KG中该轮训练的知识剔除掉
            remove_indices = torch.randperm(batch_size)
            graph.remove_edges(batch["example_idx"][remove_indices].to(self.args.device))
        graph.add_edges(list(range(graph.number_of_nodes())), list(range(graph.number_of_nodes())))
        graph.edata['relation_type'][graph.edata['relation_type'] == 0] = 1

        # create time2vec and tau: batch and graph
        batch['time2vec']=self.time_embedding(batch['start_time_list'],batch['end_time_list'])
        graph.edata['time2vec']=self.time_embedding(graph.edata['start_time_list'],graph.edata['end_time_list'])
        '''
        edata: enum*dim   batch: bsize*dim  tau: enum*bsize*dim
        edata-> unsqueeze-> num*1*dim ->repeat-> enum*bsize*dim
        注意这里每步操作的维度 
        '''  
        reverse_graph = graph.reverse(share_ndata=True, share_edata=True)
        enum=graph.number_of_edges()
        graph_input=graph.edata['time2vec'].unsqueeze(1).repeat(1,batch_size,1).view(-1, self.args.node_dim)
        batch_input=batch['time2vec'].repeat(enum,1,1).view(-1,self.args.node_dim)
        graph.edata['tau']=self.time_score( torch.cat((graph_input,batch_input),dim=1) ).view(enum,batch_size,self.args.node_dim)
        # Node and edge embedding in graph
        # 行：node/edge数目   列：batch_size   第三位:embedding_dim
        graph.ndata['h_n'] = self.node_embed(graph.ndata['node_idx']).unsqueeze(1).repeat(1, batch_size, 1) #节点向量 node_size*batch_size*embedding_size
        graph.edata['h_e'] = self.edge_embed(graph.edata['relation_type']).unsqueeze(1) #边的向量
        # prepare time order feature used in incoming_inatt_func，第二个位置是query:s_s_time s_e_time e_s_time e_e_time
        graph.edata['s_s_time'] = (graph.edata['start_time_id'].repeat(batch_size, 1) - batch['start_time_id'].unsqueeze(1)).t()        
        graph.edata['e_e_time'] = (graph.edata['end_time_id'].repeat(batch_size, 1) - batch['end_time_id'].unsqueeze(1)).t()        
        graph.edata['s_e_time'] = (graph.edata['start_time_id'].repeat(batch_size, 1) - batch['end_time_id'].unsqueeze(1)).t()        
        graph.edata['e_s_time'] = (graph.edata['end_time_id'].repeat(batch_size, 1) - batch['start_time_id'].unsqueeze(1)).t()        


        # PGNN Message Passing
        for i in range(1):  #GNN中的message passing只进行了一轮
            graph.apply_edges(func=self.incoming_inatt_func) #边增加g_e_incoming，对应文章中的mij
            graph.edata['g_e_incoming'] = graph.edata['g_e_incoming'] \
                .view(-1, batch_size, self.num_in_heads, self.in_head_dim)  # 把100维的embedding分解为num_in_heads*in_head_dim
            attn_i_incoming = (graph.ndata['h_n']    #这里用到了torch的broadcast机制，只要后个维度相同就能够点积
                               .view(-1, batch_size, self.num_in_heads, self.in_head_dim)
                               * self.PGNN_attn_i_incoming) # node_size*batch_size*head_num*head_dim，对应论文中公式（2）（3）W_K*h_j
            attn_j_incoming = (graph.edata['g_e_incoming']
                               .view(-1, batch_size, self.num_in_heads, self.in_head_dim) *
                               self.PGNN_attn_j_incoming) # edge_size*batch_size*head_num*head_dim，对应论文中公式（2）（3）W_Q*m_ij
            graph.ndata.update({'attn_self_incoming': attn_i_incoming})
            graph.edata.update({'attn_neighbor_incoming': attn_j_incoming}) # 对应论文中公式（2）（3）
            graph.apply_edges(fn.v_mul_e('attn_self_incoming', 'attn_neighbor_incoming', 'attn_neighbor_incoming')) # 对应论文中公式（2）（3） 第一个是v的，第二个是e的，第三个是输出，这里apply_edges将第三个放到e上
            attn_j_incoming = F.leaky_relu(graph.edata.pop('attn_neighbor_incoming'))   # 对应论文中公式（2）（3）
            graph.edata['a_GAT'] = edge_softmax(graph, attn_j_incoming) # 对应论文中公式（2）（2）, edge_size*batch_size*head_num*head_dim
            graph.update_all(self.incoming_msg_func, fn.sum('m', 'h_n')) # 对应论文中公式（2）（1）和公式（3），更新节点embedding
            graph.ndata['h_n'] = F.leaky_relu(graph.ndata['h_n'].view(-1, batch_size, self.args.node_dim)) # 对应论文中公式（2）（1）

        # Attention value at each step
        attn_history = []
        edge_attn_history = []

        head_indices = torch.stack((batch['head'], torch.arange(batch_size).to(self.args.device)), dim=0)
        # 初始化3.3和3.4中的g和attention分值
        graph.ndata['g_n'] = torch.zeros((graph.number_of_nodes(), batch_size, self.args.node_dim)) \
            .to(self.args.device)
        graph.ndata['g_n'][tuple(head_indices)] = graph.ndata['h_n'][tuple(head_indices)]
        graph.ndata['a'] = torch.zeros((graph.number_of_nodes(), batch_size, self.num_out_heads)) \
            .to(self.args.device)   # 初始化 node attention
        graph.ndata['a'][tuple(head_indices)] = 1   # 初始化 node attention
        
        # Prepare query vector for each example
        query = torch.cat([self.node_embed(batch['head']), self.edge_embed(batch['relation'])], dim=-1) # h_query 论文中公式（5）batch_size*embedding_dim * 2
        query = self.W_c(query) # 论文中公式（6）

        # Subgraph indices for attentive GNN
        subgraph_node_list = list(batch['head'].unsqueeze(1))   #knowledge的head节点
        subgraph_edge_list = list([] for _ in range(len(subgraph_node_list)))   # 子图中被采样的边？

        for i in range(self.args.num_step): # 对应论文3.4中Attention Flow的计算，默认参数是3
            subgraph_batch_indices = torch.cat([torch.tensor([i] * len(subgraph_node_list[i]))
                                                for i in range(len(subgraph_node_list))], dim=-1).to(self.args.device)
            subgraph_indices = torch.stack([torch.cat(subgraph_node_list, dim=-1),
                                            subgraph_batch_indices], dim=0) # 把head entities和batch id堆在一起

            graph.ndata['g_n'] = graph.ndata['g_n'].index_put(tuple(subgraph_indices), self.W_n(
                torch.cat((graph.ndata['g_n'][tuple(subgraph_indices)], query[subgraph_batch_indices, :]), dim=1))) # 对应公式（5）

            # Attention Propagation
            graph.apply_edges(func=self.outgoing_edge_func)
            attn_i_outgoing = (graph.ndata['g_n']
                               .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                               self.attn_i_outgoing)    # 第二个score
            attn_j_outgoing = (graph.edata.pop('g_e_sub_outgoing')
                               .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                               self.attn_j_outgoing)    # 第一个score
            inattn_i_outgoing = (graph.ndata['g_n']
                                 .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                                 self.inattn_i_outgoing)
            inattn_j_outgoing = (graph.edata.pop('g_e_outgoing')
                                 .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                                 self.inattn_j_outgoing)

            graph.ndata.update({'attn_i_outgoing': attn_i_outgoing, 'inattn_i_outgoing': inattn_i_outgoing})
            graph.edata.update({'attn_j_outgoing': attn_j_outgoing, 'inattn_j_outgoing': inattn_j_outgoing})
            graph.apply_edges(fn.u_dot_e('attn_i_outgoing', 'attn_j_outgoing', 'tau_attn')) # 3.4中T_ij对应的第1个score？
            graph.apply_edges(fn.u_dot_e('inattn_i_outgoing', 'inattn_j_outgoing', 'tau_inattn'))   # 3.4中T_ij对应的第2个score？
            tau = F.leaky_relu(graph.edata.pop('tau_attn')) + F.leaky_relu(graph.edata.pop('tau_inattn'))   # 3.4中计算T_ij

            graph.edata['transition'] = edge_softmax(reverse_graph, tau).squeeze(-1)    # 计算T_ij
            prev_a = graph.ndata['a'].mean(2)   #
            graph.apply_edges(fn.u_mul_e('a', 'transition', 'a_tilde')) # 3.4节一开始的公式
            graph.update_all(fn.copy_e('a_tilde', 'a_tilde'), fn.sum('a_tilde', 'a'))
            edge_attn_history.append(graph.edata['a_tilde'][:-graph.number_of_nodes()].mean(2))
            # 'a': (num_nodes, batch_size, num_att_heads)

            # Subgraph Sampling
            subgraph, subgraph_node_list, subgraph_edge_list = self.sample_subgraph(graph, prev_a,
                                                                                    graph.edata['a_tilde'].mean(2),
                                                                                    subgraph_node_list,
                                                                                    subgraph_edge_list)

            # SGNN Message Passing
            subgraph.apply_edges(func=self.incoming_att_func)
            subgraph.edata['g_e_incoming'] = subgraph.edata['g_e_incoming'] \
                .view(-1, self.num_out_heads, self.out_head_dim)
            attn_i_incoming = (subgraph.ndata['g_n'].view(-1, self.num_in_heads, self.in_head_dim) *
                               self.SGNN_attn_i_incoming)
            attn_j_incoming = (subgraph.edata['g_e_incoming'] *
                               self.SGNN_attn_j_incoming)
            subgraph.ndata.update({'attn_i_incoming': attn_i_incoming})
            subgraph.edata.update({'attn_j_incoming': attn_j_incoming})
            subgraph.apply_edges(fn.v_mul_e('attn_i_incoming', 'attn_j_incoming', 'attn_j_incoming'))

            attn_j_incoming = F.leaky_relu(subgraph.edata.pop('attn_j_incoming'))
            subgraph.edata['a_GAT'] = edge_softmax(subgraph, attn_j_incoming)
            subgraph.update_all(self.incoming_msg_func, fn.sum('m', 'g_n'))

            subgraph.ndata['g_n'] = subgraph.ndata['g_n'].view(-1, self.args.node_dim)
            subgraph.ndata['g_n'] += subgraph.ndata['a'].mean(1, keepdim=True) * self.W_h(subgraph.ndata['h_n'])
            subgraph.ndata['g_n'] = F.leaky_relu(subgraph.ndata['g_n'])

            for sub_idx, sub_g in enumerate(dgl.unbatch(subgraph)):
                graph.ndata['g_n'] = graph.ndata['g_n'].index_put((sub_g.ndata['node_idx'], torch.tensor(sub_idx)),
                                                                  sub_g.ndata['g_n'])

            attn_history.append(graph.ndata['a'].mean(2))
            # print("a:",graph.ndata['a'].mean(2))


        return attn_history

    def sample_subgraph(self, graph, prev_a, a, prev_subgraph_nodes, prev_subgraph_edges):
        """Given node / edge attention distribution, sample subgraph at each step"""
        new_subgraph_nodes = []
        new_subgraph_edges = []
        new_subgraphs = []
        sample_from = [torch.topk(prev_a[:, i],
                                  dim=0, k=min(self.args.num_sample_from, len(prev_subgraph_nodes[i])))[1]
                       for i in range(len(prev_subgraph_nodes))]    # 每个subgraph的采样节点池 最多采num_sample_from个节点

        for i, sample_pool in enumerate(sample_from):   # 遍历batch中每个子图
            edges = tuple({edge for query_node in sample_pool for edge
                           in np.random.permutation(graph.out_edges(query_node, form='eid').cpu())
                           [:self.args.max_num_neighbor].tolist()}) # query head的邻接边
            topk_edges = torch.tensor(edges)[torch.topk(
                a[edges, i], dim=0, k=min(len(edges), self.args.max_num_neighbor))[1]].to(a.device) # 选取query head的邻接边中attention分值top k的边集合

            if len(prev_subgraph_edges[i]) > 0:
                topk_edges = torch.cat([prev_subgraph_edges[i], topk_edges], dim=-1)
            new_subgraph = graph.edge_subgraph(topk_edges)  #
            new_subgraph.ndata['node_idx'] = new_subgraph.parent_nid.to(a.device)
            new_subgraph.edata['edge_idx'] = new_subgraph.parent_eid.to(a.device)
            new_subgraph.ndata['g_n'] = graph.ndata['g_n'][new_subgraph.parent_nid][:, i]
            new_subgraph.ndata['a'] = graph.ndata['a'][new_subgraph.parent_nid][:, i]
            new_subgraph.ndata['h_n'] = graph.ndata['h_n'][new_subgraph.parent_nid][:, i]
            new_subgraph.edata['h_e'] = graph.edata['h_e'][new_subgraph.parent_eid].squeeze(1)
            # time features
            new_subgraph.edata['s_s_time'] = graph.edata['s_s_time'][new_subgraph.parent_eid][:, i]
            new_subgraph.edata['e_e_time'] = graph.edata['e_e_time'][new_subgraph.parent_eid][:, i]
            new_subgraph.edata['s_e_time'] = graph.edata['s_e_time'][new_subgraph.parent_eid][:, i]
            new_subgraph.edata['e_s_time'] = graph.edata['e_s_time'][new_subgraph.parent_eid][:, i]
            new_subgraph.edata['tau'] = graph.edata['tau'][new_subgraph.parent_eid][:, i]
            new_subgraphs.append(new_subgraph)
            new_subgraph_nodes.append(new_subgraph.ndata['node_idx'])
            new_subgraph_edges.append(new_subgraph.edata['edge_idx'])

        return dgl.batch(new_subgraphs), new_subgraph_nodes, new_subgraph_edges

    def outgoing_edge_func(self, edges):
        """Attention propagation message computation"""
        return {
            'g_e_sub_outgoing': edges.dst['g_n'] + edges.data['h_e'] + edges.data['tau'],   # 第一个score
            'g_e_outgoing': edges.dst['h_n'] + edges.data['h_e'] + edges.data['tau']   # 第二个score
        }

    def incoming_inatt_func(self, edges):
        """PGNN message computation 对应论文中公式1"""
        # edges.src['h_n']是源node的h_n feature
        translational = edges.src['h_n'] + edges.data['h_e'] + edges.data['tau']    #m_ij的计算
        s_s_past = self.inattn_s_s_past_lin(translational).masked_fill((edges.data['s_s_time'] >= 0).unsqueeze(-1), 0)  # 都是边数目*batch_size*
        s_s_present = self.inattn_s_s_present_lin(translational).masked_fill((edges.data['s_s_time'] != 0).unsqueeze(-1), 0)
        s_s_future = self.inattn_s_s_future_lin(translational).masked_fill((edges.data['s_s_time'] <= 0).unsqueeze(-1), 0)

        e_e_past = self.inattn_e_e_past_lin(translational).masked_fill((edges.data['e_e_time'] >= 0).unsqueeze(-1), 0)  # 都是边数目*batch_size*
        e_e_present = self.inattn_e_e_present_lin(translational).masked_fill((edges.data['e_e_time'] != 0).unsqueeze(-1), 0)
        e_e_future = self.inattn_e_e_future_lin(translational).masked_fill((edges.data['e_e_time'] <= 0).unsqueeze(-1), 0)
        # g_e_incoming返回到边上
        return {
            'g_e_incoming': s_s_past + s_s_present + s_s_future + e_e_past + e_e_present + e_e_future
        }

    def incoming_att_func(self, edges):
        """SGNN message computation"""
        translational = edges.src['g_n'] + edges.data['h_e'] + edges.data['tau']    #m_ij的计算
        s_s_past = self.attn_s_s_past_lin(translational).masked_fill((edges.data['s_s_time'] >= 0).unsqueeze(-1), 0)  # 都是边数目*batch_size*
        s_s_present = self.attn_s_s_present_lin(translational).masked_fill((edges.data['s_s_time'] != 0).unsqueeze(-1), 0)
        s_s_future = self.attn_s_s_future_lin(translational).masked_fill((edges.data['s_s_time'] <= 0).unsqueeze(-1), 0)

        e_e_past = self.attn_e_e_past_lin(translational).masked_fill((edges.data['e_e_time'] >= 0).unsqueeze(-1), 0)  # 都是边数目*batch_size*
        e_e_present = self.attn_e_e_present_lin(translational).masked_fill((edges.data['e_e_time'] != 0).unsqueeze(-1), 0)
        e_e_future = self.attn_e_e_future_lin(translational).masked_fill((edges.data['e_e_time'] <= 0).unsqueeze(-1), 0)
        # g_e_incoming返回到边上
        return {
            'g_e_incoming': s_s_past + s_s_present + s_s_future + e_e_past + e_e_present + e_e_future
        }

    def incoming_msg_func(self, edges):
        return {'m': (edges.data['g_e_incoming'] * edges.data['a_GAT'])}

    def time_embedding(self,start_time,end_time):
        '''Given start_time_list and end_time_list, create the time_embedding'''
        start_gra=[]
        end_gra=[]
        batch_size=start_time.shape[0]
        for i in range(self.args.granularity_dim):
            start_gra.append(self.start_t2v[i](start_time[:,i].view(-1,1)))
            end_gra.append(self.end_t2v[i](end_time[:,i].view(-1,1)))
        start_time=torch.cat(start_gra,dim=1)
        end_time=torch.cat(end_gra,dim=1)
        # print("start_time shape: ",start_time.shape)
        # print("end_time shape: ",end_time.shape)
        # print("time shape: ",torch.cat((start_time, end_time),dim=-1).shape)
        # exit(0)
        time = torch.cat((start_time, end_time),dim=-1)
        time = self.time_linear(time)
        return time


class NodeEmbedding(nn.Module):
    def __init__(self, args):
        super(NodeEmbedding, self).__init__()
        self.args = args
        self.node_dim = args.node_dim
        self.diachronic_dim = int(args.node_dim * args.gamma)

        self.synchronic_embed = nn.Embedding(len(args.entity_vocab), self.node_dim, padding_idx=0)
        self.diachronic_embed = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)
        self.diachronic_w = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)
        self.diachronic_b = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)

    def forward(self, indices, time_indices=None, diachronic=False):
        """
        :param indices: (num_nodes,)
        :param time_indices: (num_nodes,)
        :param diachronic: bool, True to get diachronic embedding
        :return:
        """
        node_embed = self.synchronic_embed(indices)  # (num_nodes, node_dim)

        if diachronic:
            node_embed[:, self.diachronic_dim:] += \
                self.diachronic_embed(indices) * \
                torch.sin(self.diachronic_w(indices) * time_indices.unsqueeze(1) + self.diachronic_b(indices))

        return node_embed

    def time_transform(self, node_embed, indices, time_indices, masked=False):
        """
        Given embedding, transform to diachronic embedding
        :param node_embed: (num_nodes, node_dim) or (num_nodes, batch_size, node_dim)
        :param indices: (num_nodes,)
        :param time_indices:  (num_nodes,)
        :param masked: whether to mask nodes not included in subgraph
        :return:
        """
        diachronic_embed = \
            self.diachronic_embed(indices).unsqueeze(1) * \
            torch.sin(self.diachronic_w(indices).unsqueeze(1).repeat(1, 16, 1) * time_indices.unsqueeze(2)
                      + self.diachronic_b(indices).unsqueeze(1).repeat(1, 16, 1))

        if node_embed.dim() == 2:
            # for incoming attention
            mask = (torch.sum(node_embed, dim=-1) == 0)
            node_embed[:, self.diachronic_dim:] += diachronic_embed
        else:
            # for outgoing attention
            mask = (torch.sum(node_embed, dim=-1) == 0).unsqueeze(-1)
            node_embed[:, :, self.diachronic_dim:] += diachronic_embed

        if masked:
            node_embed.masked_fill(mask, 0)
        return node_embed
