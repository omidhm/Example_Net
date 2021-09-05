# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:36:14 2020

@author: Omid
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from worker import worker_new
import matplotlib.pyplot as plt
from time_Kfast import t
import copy
from master_loop import Get_Res_Worker
from master_loop import Get_Res_Worker_Kfast


for i in range(10):
    
    learning_rate = 0.01
    untrained_model_name= 'untrained_model.pkl'
    trained_model_name= 'trained_model.pkl'
    merged_model_name= 'merged_model.pkl'
    ModelList = []          ## list for models
    Res = []
    MaxItr = 100
    MaxTime = 100
    batch_size = 2 ^ (i+1)
    smooth = 500
    VGG = 2
    CUDAA = 1
    
    
    n_workers = 10
    Avg = n_workers
    Asking = [5,8,10]
    
    
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
                 
    
                     
    #Split the test sets                            
    #testSize = 1000                                      
    #TestSet1, TestSet2 = torch.utils.data.random_split(trainset, [testSize,(len(trainset)-testSize)])
    TestSet1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,  transform=transform)
                                        
    testloader = torch.utils.data.DataLoader(TestSet1, batch_size=200, shuffle=False, num_workers=0)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #############################################################################
    if VGG == 0:
        
        class model(nn.Module):
            def __init__(self):
                super(model, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
        
            def forward(self, x):
                x = self.pool(nn.functional.relu(self.conv1(x)))
                x = self.pool(nn.functional.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = nn.functional.relu(self.fc1(x))
                x = nn.functional.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    #############################################################################
    elif VGG == 1:
        VGG_types = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        
        class model(nn.Module):
            def __init__(self, in_channels=3, num_classes=1000):
                super(model, self).__init__()
                self.in_channels = in_channels
                self.conv_layers = self.create_conv_layers(VGG_types['VGG19'])
                
                self.fcs = nn.Sequential(
                    nn.Linear(512*7*7, 4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, num_classes)
                    )
                
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.reshape(x.shape[0], -1)
                x = self.fcs(x)
                return x
        
            def create_conv_layers(self, architecture):
                layers = []
                in_channels = self.in_channels
                
                for x in architecture:
                    if type(x) == int:
                        out_channels = x
                        
                        layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                             kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                   nn.BatchNorm2d(x),
                                   nn.ReLU()]
                        in_channels = x
                    elif x == 'M':
                        layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                        
                return nn.Sequential(*layers)
    #############################################################################
    
    elif VGG == 2:
        class model(nn.Module):
            def __init__(self, num_classes=10):
                super(model, self).__init__()
                self.conv1 = torch.nn.Sequential()
                self.conv1.add_module("conv1", nn.Conv2d(3, 32, 3, 1, 2))
                self.conv1.add_module("relu1", torch.nn.ReLU())
                self.conv2 = torch.nn.Sequential()
                self.conv2.add_module("conv2", nn.Conv2d(32, 64, 3, 1, 2))
                self.conv2.add_module("relu2", torch.nn.ReLU())
                self.conv2.add_module("pool2", torch.nn.MaxPool2d(2,2))
                self.drop1 = torch.nn.Dropout(0.25)
        
                self.conv3 = torch.nn.Sequential()
                self.conv3.add_module("conv3", nn.Conv2d(64, 128, 3, 1, 2))
                self.conv3.add_module("relu3", torch.nn.ReLU())
                self.conv3.add_module("pool3", torch.nn.MaxPool2d(2))
        
                self.conv4 = torch.nn.Sequential()
                self.conv4.add_module("conv4", nn.Conv2d(128, 128, 3, 1, 2, padding_mode='SAME'))
                self.conv4.add_module("relu4", torch.nn.ReLU())
                self.conv4.add_module("pool4", torch.nn.MaxPool2d(2))
                self.drop2 = torch.nn.Dropout(0.25)
        
                self.dense1 = torch.nn.Sequential()
                self.dense1.add_module("fc1", nn.Linear(128 * 6 * 6, 1500))
                self.dense2 = torch.nn.Sequential()
                self.drop3 = torch.nn.Dropout(0.5)
        
                self.dense2.add_module("fc2", nn.Linear(1500, num_classes))
                
            def forward(self,x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(conv1)
                conv2 = self.drop1(conv2)
                conv3 = self.conv3(conv2)
                conv4 = self.conv4(conv3)
                conv4 = self.drop2(conv4)
                conv4 = conv4.reshape(conv4.size(0), -1)
                # print(conv2.shape)
                fc1 = self.dense1(conv4)
                fc1 = self.drop3(fc1)
                fc2 = self.dense2(fc1)
                return fc2
    elif VGG == 3:
    #    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    #    Fix_net = model.to(device) 
    #    net = model.to(device)
        pass       
    #############################################################################
    from torchvision import datasets, models, transforms
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    #net = model().to(device)
    
    #if CUDAA == 0:
    #    Fix_net = model()
    #    net = model()
    #else:   
    #    Fix_net = model().to(device) 
    #    net = model().to(device)
        
        
    #Fix_net_state_dict = copy.deepcopy(Fix_net.state_dict())
    #Fix_net.load_state_dict(Fix_net_state_dict)
        
    #model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    #model = models.vgg16_bn()
    
    
    ############################# VVV  ###################################
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
    
    ######################### TTTT ###############################
    #Fix_net = model()
    #Fix_net_par1 = Fix_net.named_parameters()
    #Fix_net_par = copy.deepcopy(dict(Fix_net_par1))
    #Fix_net.load_state_dict(Fix_net.state_dict())
    ##############################################################
        
    
    #Make fix model for start
    #Fix_net = model()
    # Start
    #upload the net parameters to the net()
    #net = model()
    #optimizer = torch.optim.SGD(net.classifier.parameters(), lr = learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Run algorithm for different Scenarios + Plot Figures 
    Err_Storage = [[] for i in range(Avg+1)]
    Time_Storage = [[] for i in range(Avg+1)]
    for i in Asking:
    #for i in [5]:
        Kfast = i
    #    Fix_net_par1 = Fix_net.named_parameters()
    #    Fix_net_par = copy.deepcopy(dict(Fix_net_par1))
    #    Fix_net_state_dict = copy.deepcopy(Fix_net.state_dict())
        Fix_net_state_dict = copy.deepcopy(Fix_net.state_dict())
        param_master_loop = [Kfast, Avg, learning_rate, Fix_net_state_dict,
                             worker_data_loader, testloader, net, net_woker,MaxItr,
                             MaxTime, smooth, criterion, VGG, CUDAA]
        Err_Storage[i-1], Time_Storage[i-1] = Get_Res_Worker(param_master_loop)
        
        if cnt > 0:
            if min(Err_Storage[i-1]) > min(Err_Storage[i-2]):
                break
        cnt += 1
    
    
    
    ESS  = [x for x in Err_Storage if x]   
    TSS = [x for x in Time_Storage if x] 

### Kfast ##########################

#Err_Storage = Err_Storage[:len(Err_Storage)-1]
#Time_Storage = Time_Storage[:len(Time_Storage)-1]
#determine = 0
#Checking_idea = []
#Fix_net_state_dict = copy.deepcopy(net.state_dict())
#param_master_loop = [1, Avg, learning_rate, Fix_net_par, 
#                     worker_data_loader, testloader, net, MaxItr,
#                     MaxTime, smooth, optimizer, criterion,  VGG, CUDAA]
#E_Kfast, T_Kfast, point_Err, point_time = Get_Res_Worker_Kfast(param_master_loop)
#Err_Storage[Avg]  = E_Kfast
#Time_Storage[Avg] = T_Kfast
#Asking.append(Avg+1)
####################################
#Err_Storage = []
#Time_Storage = []
#color = -1

#
color = -1
for i in Asking:
    color += 1
    color_1 = ['b-','r-','c-','k-','g-']
#    color_2 = ['b','r','c','k','g']
#    Label = [i for i in range(len(Err_Storage)-1)]
#    Label.append('Kfast')
    for j in range(len(Err_Storage[i-1])-1):
#        plt.plot(Time_Storage[i][j] ,Err_Storage[i][j] ,color_1[i],
#                (Time_Storage[i][j], Time_Storage[i][j+1]), 
#                (Err_Storage[i][j] , Err_Storage[i][j+1]), color_2[i])
        plt.plot(Time_Storage[i-1] ,Err_Storage[i-1], color_1[color], label=i if j == 0 else "")
#        
#for i in range(np.size(point_Err)):
#     plt.plot(point_time[i], point_Err[i],'go')        
#     
plt.legend()
plt.ylabel('CrossEntropyLoss') 
plt.xlabel('Clock_Time')  
plt.title('Err Base Time')


#for i in range(len(Err_Storage)):
#    print(i)
#    color_1 = ['b-','r-','c-','k-','g-']
#    color_2 = ['b','r','c','k','g']
#    Label = [Asking[i] for i in range(len(Err_Storage)-1)]
#    Label.append('Kfast')
#    for j in range(len(Err_Storage[i])-1):
##        plt.plot(Time_Storage[i][j] ,Err_Storage[i][j] ,color_1[i],
##                (Time_Storage[i][j], Time_Storage[i][j+1]), 
##                (Err_Storage[i][j] , Err_Storage[i][j+1]), color_2[i])
#        plt.plot(Time_Storage[i][j] ,Err_Storage[i][j] ,color_1[i],
#                (Time_Storage[i][j], Time_Storage[i][j+1]), 
#                (Err_Storage[i][j] , Err_Storage[i][j+1]), color_2[i],
#                label=Label[i] if j == 0 else "")
## 
#for i in range(np.size(point_Err)):
#     plt.plot(point_time[i], point_Err[i],'ko')
#     
#plt.legend()
#plt.ylabel('CrossEntropyLoss') 
#plt.xlabel('Clock_Time')  
#plt.title('Err Base Time')


#
#for i in range(len(Err_Storage)):
#    print(i)
#    color_1 = ['b-','r-']
#    color_2 = ['b','r']
#    Label = ['1','2']
##    for j in range(len(Time_Storage[len(Err_Storage)-1])-1):
#    for j in range(10):
#        plt.plot(Err_Storage[i])
#plt.legend()
#plt.ylabel('CrossEntropyLoss') 
#plt.xlabel('Clock_Time')  
#plt.title('Err Base Itr')

