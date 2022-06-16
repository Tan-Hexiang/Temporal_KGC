import torch
import torch.nn as nn
class T2Vlayer(nn.Module):
    def __init__(self,output_dim) -> None:
        '''
        granularity_dim为时间维度，如[year,month,day]维度为3
        '''
        super().__init__()
        assert output_dim > 1
        self.output_dim=output_dim
        self.w=nn.Parameter(torch.randn(1),requires_grad=True)
        self.p=nn.Parameter(torch.randn(1),requires_grad=True)
        self.fre=nn.Parameter(torch.randn(1,output_dim-1),requires_grad=True)
        self.phi=nn.Parameter(torch.randn(1,output_dim-1),requires_grad=True)

        nn.init.xavier_uniform_(self.fre)
        nn.init.xavier_uniform_(self.phi)

    def forward(self,x):
        '''
        x为（batch,1)
        output: time2vec bsize个张量(),第一维为时间序列特征，其余output_dim-1维为时间周期特征
        参数有： 序列参数w x+ b  output_dim-1个周期参数 sine(fre*x+phi)
        '''
        seqfeature=self.w*x+self.p  #(bsize,1)
        periodfeature=self.activate_sin_func(torch.mul(x,self.fre)+self.phi)    #(bsize,1) *(1,output_dim-1) =(bsize,ouput_dim-1)
        return torch.cat((seqfeature,periodfeature),dim=1)  #(bsize,output_dim)

    def activate_sin_func(self,x):
        return torch.sin(x)

