import argparse
from secrets import choice
from pandas import test

from sklearn import datasets
from params import Params
from trainer import Trainer
from tester import Tester
from dataset import myDataset

parser=argparse.ArgumentParser(description='Temporal KG Entity Prediction')
parser.add_argument('-datapath',help='The prepath of the dataset',type=str,default='/data/tanhexiang/temporal_link_prediction/data/',choices=['/data/tanhexiang/temporal_link_prediction/data/'])
parser.add_argument('-dataset_name',help='Dataset name',type=str,default='wikidata12k',choices=['wikidata12k','yago11k','yago4'])
parser.add_argument('-model',help='Model name',type=str,default='TransE', choices=['TransE'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=500, choices = [500])
parser.add_argument('-bsize', help='Batch size', type=int, default=512, choices = [512])
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001, choices = [0.001])
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=100, choices = [100])
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4, choices = [0.0, 0.2, 0.4])
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=20, choices = [20])
parser.add_argument('-valid_ratio',help='split data of this ratio to validation',type=float,default=0.1)
parser.add_argument('-test_ratio',help='split data of this ratio to test',type=float,default=0.1)
parser.add_argument('-granularity_dim',help='the granularity of the time',type=int,default=3,choices=[3,1])

args=parser.parse_args()
params=Params(  ne=args.ne, 
    bsize=args.bsize, 
    lr=args.lr, 
    reg_lambda=args.reg_lambda, 
    emb_dim=args.emb_dim, 
    neg_ratio=args.neg_ratio, 
    dropout=args.dropout, 
    save_each=args.save_each, 
    valid_ratio=args.valid_ratio,
    test_ratio=args.test_ratio,
    granularity_dim=args.granularity_dim
    )
# create dataset
dataset=myDataset(
    datapath=args.datapath,
    dataname=args.dataset_name,
    granularity_dim=args.granularity_dim,
    valid_ratio=args.valid_ratio,
    test_ratio=args.test_ratio
    )
print("Create dataset:\n\t in  ",args.datapath+args.dataset_name,"\n\tgranularity_dim:",args.granularity_dim)

# create trainer and call the main function of training. model saved in file
trainer=Trainer(dataset=dataset, params=params, model_name=args.model)
print("Creat Trainer, call train()")
# trainer.train()


# test the trained model and find the best one 
print("Start to test the MRR of the model")
validation_idx = [str(int(args.save_each * (i + 1))) for i in range(args.ne // args.save_each)]
best_mrr = -1.0
best_index = '0'
model_prefix = "models/" + args.model + "/" + args.dataset_name + "/" + params.str_() + "_"

for idx in validation_idx:
    model_path = model_prefix + idx + ".chkpnt"
    tester = Tester( model_path,dataset,test_or_valid='valid')
    mrr = tester.test()
    if mrr > best_mrr:
        best_mrr = mrr
        best_index = idx

# testing the best chosen model on the test set
print("Best epoch: " + best_index)
model_path = model_prefix + best_index + ".chkpnt"
tester = Tester(dataset, model_path, "test")
tester.test()
