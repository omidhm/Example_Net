# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:36:14 2020

@author: Omid
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
from master_loop import Get_Res_Worker

learning_rate = 0.0005
untrained_model_name= 'untrained_model.pkl'
trained_model_name= 'trained_model.pkl'
merged_model_name= 'merged_model.pkl'
ModelList = []          ## list for models
Res = []
MaxItr = 100
MaxTime = 10
batch_size = 20
smooth = 500
VGG = 2
CUDAA = 1


n_workers = 1000
Avg = n_workers
Asking = [2]


transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,  transform=transform)
temp = tuple([len(trainset)//n_workers for i in range(n_workers)])

dataset = torch.utils.data.random_split(trainset, temp)
worker_data_loader = []
for data_w in dataset:
    worker_data_loader.append(torch.utils.data.DataLoader(data_w, batch_size=batch_size, shuffle=True))

TestSet1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,  transform=transform)
                                    
testloader = torch.utils.data.DataLoader(TestSet1, batch_size=200, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


############################# VVV  ###################################
from torchvision import models, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = models.vgg11(pretrained=True)
model2 = models.vgg11(pretrained=True)
model3 = models.vgg11(pretrained=True)
net = model1.to(device)
Fix_net  = model2.to(device)
net_woker = model3.to(device)

net.classifier[6].out_features = 10
net_woker.classifier[6].out_features = 10
Fix_net.classifier[6].out_features = 10

for param in net.features.parameters():
    param.requires_grad = False
for param in net_woker.features.parameters():
    param.requires_grad = False
for param in Fix_net.features.parameters():
    param.requires_grad = False
##############################################################
criterion = nn.CrossEntropyLoss()

# Run algorithm for different Scenarios + Plot Figures 
Err_Storage = [[] for i in range(Avg+1)]
Time_Storage = [[] for i in range(Avg+1)]

Kfast = Asking[0]
Fix_net_state_dict = copy.deepcopy(Fix_net.state_dict())
param_master_loop = [Kfast, Avg, learning_rate, Fix_net_state_dict,
                         worker_data_loader, testloader, net, net_woker,MaxItr,
                         MaxTime, smooth, criterion, VGG, CUDAA]
Err_Storage[0], Time_Storage[0] = Get_Res_Worker(param_master_loop)


color = 0
color_1 = ['b-','r-','c-','k-','g-']
plt.plot(Time_Storage[0] ,Err_Storage[0], color_1[color], label=i if j == 0 else "")   
plt.legend()
plt.ylabel('CrossEntropyLoss') 
plt.xlabel('Clock_Time')  
plt.title('Err Base Time')



