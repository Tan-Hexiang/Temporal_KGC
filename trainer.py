from ast import Param
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import myDataset
from model import mymodel
from transE import TransE

class Trainer:
    def __init__(self,dataset,params,model_name) ->None:
        self.model_name=model_name
        instance_gen = globals()[model_name]
        self.params=params
        self.dataset=dataset
        self.model = nn.DataParallel(instance_gen(dataset=self.dataset, params=params))
    
    def train(self, early_stop=False):
        # model 设为train模式
        self.model.train()
        
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), 
        #     lr=self.params.lr, 
        #     weight_decay=self.params.reg_lambda
        # ) #weight_decay corresponds to L2 regularization
        optimizer = torch.optim.Adam(
            [ param for param in self.model.parameters() if param.requires_grad == True], 
            lr=self.params.lr, 
            weight_decay=self.params.reg_lambda
        ) #weight_decay corresponds to L2 regularization


        loss_f = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start = time.time()
            
            while not last_batch:
                optimizer.zero_grad()
                
                heads, rels, tails, start_time, end_time = self.dataset.nextBatch(self.params.bsize, neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()
                
                scores = self.model(heads, rels, tails, start_time, end_time)
                
                ###Added for softmax####
                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio+1)
                # zero是target位置，这里score第一列是pos sample
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                
            print(time.time() - start)
            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.model_name + "," + self.dataset.name + ")")
            
            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "/data/tanhexiang/temporal_link_prediction/models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")