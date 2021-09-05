# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:02:45 2020

@author: Omid
"""
def Kfast_Func(net, Net_Kfast, Checking_idea):
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
    from Kfast import Kfast_Func
    
    t1 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A1 = 0
    A2 = 0
    A4 = 0
    
    Temp_cnt = 0
    Temp_Sum = 0
    for p in zip(net.parameters(), Net_Kfast.parameters()):
        Where = len(p[0].grad.size())
        Temp_cnt += 1
        F = 0
        if Where == 1:
            A1 = torch.matmul(p[0].grad.to(device), p[1].grad.to(device))  
#            print(torch.sum(torch.abs(p[0].grad)), torch.sum(torch.abs(p[1].grad)))
#            print('A1', A1)
        elif Where == 2:
            A2 = torch.trace(torch.matmul(p[0].grad.to(device), p[1].grad.T.to(device)))
#            print(torch.diag(torch.matmul(p[0].grad, p[1].grad.T)))
            F += torch.trace(torch.matmul(p[0].grad.to(device), p[1].grad.T.to(device)))
            print(torch.sum((p[0].grad.to(device))), torch.sum((p[1].grad.to(device))))
            if F>1:
                
#                print(F)
                Checking_idea.append(A2)
#            print('torch.matmul(p[0].grad, p[1].grad.T)', torch.sum(torch.matmul(p[0].grad, p[1].grad.T),0))
#            print('A2', torch.sum(torch.diag(torch.matmul(p[0].grad, p[1].grad.T))))
            
        elif Where == 4:
            Conv_test_trace = torch.matmul(p[0].grad.to(device), p[1].grad.to(device)) 
            for i in range(p[0].grad.size()[0]):
                for j in range(p[0].grad.size()[1]):
                    A4 = torch.sum(torch.trace(Conv_test_trace[i][j]))
                    
    print(time.time()  - t1)
    return Checking_idea


#                    print('A4', A4)
#        if A2<0:
#            print('A1', A1, 'A2', A2, 'A4', A4)
#        Temp_Sum += A1 + A2 + A4
#    print('Temp_cnt', Temp_cnt, Temp_Sum)
    
####
#len(torch.randn(400).size())
#####
#tensor1 = torch.randn(400,120)
#tensor2 = torch.randn(400,120).T
#torch.matmul(tensor1, tensor2)
#torch.matmul(tensor1, tensor2).size()
#torch.trace(torch.matmul(tensor1, tensor2))
#####
#####
#tensor1 = torch.randn(6,3,5,5)
#tensor1.size()[0]
#tensor2 = torch.randn(6,3,5,5)
#torch.matmul(tensor1, tensor2)[1][0]
#torch.matmul(tensor1, tensor2).size()
#Conv_test_trace = torch.matmul(tensor1, tensor2)
#for i in range(tensor1.size()[0]):
#    for j in range(tensor1.size()[1]):
#        print(torch.sum(torch.trace(Conv_test_trace[i][j])))
##### 
#####
#tensor1 = torch.randn(6)
#tensor2 = torch.randn(6)
#torch.matmul(tensor1, tensor2)
#torch.matmul(tensor1, tensor2).size()
#torch.matmul(tensor1, tensor2)
####