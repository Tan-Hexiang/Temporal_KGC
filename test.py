import torch
# a=torch.ones((2,1)).repeat(5,1)
# b=torch.ones((10,1))
# print(a-b)

a=[]
for i in range(10):
    a.append([1])
print(a)
print(torch.tensor(a).shape)