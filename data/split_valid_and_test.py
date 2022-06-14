from fileinput import filename
import torch
import random

filenames=["./wikidata12k/","./yago4/","./yago11k/"]
test_ratio=0.1
valid_ratio=0.2


for file in filenames:
    with open(file+'temporal','r') as f:
        data=f.readlines()
        random.shuffle(data)
        l=len(data)
        test_point=int(l*test_ratio)
        valid_point=int(l*valid_ratio)
        test_data=data[:test_point]
        valid_data=data[test_point:valid_point]
        train_data=data[valid_point:]
    with open(file+'train.txt','w') as train, open(file+'test.txt','w') as test,open(file+'valid.txt','w') as valid:
        train.writelines(train_data)
        test.writelines(test_data)
        valid.writelines(valid_data)

