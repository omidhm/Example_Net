# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:33:45 2020

@author: Omid
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:02:45 2020

@author: Omid
"""
def Kfast_Func_new(G1, G2, parameter_Kfast):
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
    
    [countNegative, countItr, Kfast, datasize, Avg] = parameter_Kfast 
    
    burnin = .01 * datasize * Kfast
    step = 1
    thresh = 10
    
    t1 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    A1 = 0
    A2 = 0
    A4 = 0
    
    Temp_cnt = 0
    Temp_Sum = 0
    i = 0
    pos = 0
    neg = 0
    
    for l1 in zip(G1, G2):
        Where = len(l1[0].shape)
#        print(Where)
        Temp_cnt += 1
        F = 0
        if Where == 1:
            i += 1
            A1 = torch.matmul(l1[0].to(device), l1[1].to(device)) 
#            print(A1)
            if A1 > 0:
                pos += 1
            else:
                neg += 1
        elif Where == 2:
            A2 = torch.trace(torch.matmul(l1[0].to(device), l1[1].T.to(device)))
#            print(A2)
            if A2 > 0:
#                print(A2, pos)
                pos += 1
            else:
                neg += 1
        elif Where == 4:
            Conv_test_trace = torch.matmul(l1[0].to(device), l1[1].to(device)) 
            for i in range(l1[0].size()[0]):
                for j in range(l1[0].size()[1]):
                    A4 = torch.sum(torch.trace(Conv_test_trace[i][j]))
                    if A4 > 0:
#                        print(A4, pos)
                        pos += 1
                    else:
                        neg += 1
#    print('neg', neg, 'pos', pos) 
    # it is different from kfast.
    # here we have more nodes which was one node in Kfast       
    if neg > pos:
        countNegative += 1
    if pos > neg:
        countNegative -= 1
    
    if countNegative > thresh and countItr > burnin and Kfast <= Avg - step:
        print(countItr, burnin)
        Kfast = Kfast + step
        countNegative = 0
        countItr = 0
    
    countItr += 1
    res_Kfast = [countNegative, countItr, Kfast]
#        print(pos, neg)
#        print('counter', counter, 'pos', pos, 'neg', neg)
#        print(A1)
    print((time.time()  - t1)/60, 'min')
    
    return res_Kfast
