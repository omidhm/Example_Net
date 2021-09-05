# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:36:54 2020

@author: Omid
"""
from master_loop import Get_Res_Worker_Kfast
from Kfast_new import Kfast_Func_new
import copy
Asking = ['Kfast']
Kfast = 1
Avg = 20
MaxTime = 2000
MaxItr = 2000
smooth = 100
learning_rate = .01
E_test = copy.deepcopy(Err_Storage)
T_test = copy.deepcopy(Time_Storage)
determine = 0
Checking_idea = []
E_test = []
T_test = []


Fix_net_par1 = Fix_net.named_parameters()
Fix_net_par = copy.deepcopy(dict(Fix_net_par1))
param_master_loop = [Kfast, Avg, learning_rate, Fix_net_par, worker_data_loader, testloader, net, MaxItr, MaxTime, smooth, determine, Checking_idea]
E_Kfast, T_Kfast = Get_Res_Worker_Kfast(param_master_loop)
E_test.append(E_Kfast)
T_test.append(T_Kfast)

plt.plot(Checking_idea[0:1000])     
min(Checking_idea)



for i in range(len(T_test)):
    print(i)
    color_1 = ['b-','r-','c-','k-','g-','m-']
    color_2 = ['b','r','c','k','g','m']
    Label = [Asking[i] for i in range(len(E_test))]
    for j in range(len(E_test[i])-1):
        plt.plot(T_test[i][j] ,E_test[i][j] ,color_1[i],
                (T_test[i][j], T_test[i][j+1]), 
                (E_test[i][j] , E_test[i][j+1]), color_2[i],
                label=Label[i] if j == 0 else "")
plt.legend()
plt.ylabel('CrossEntropyLoss') 
plt.xlabel('Clock_Time')  
plt.title('Err Base Time')

Label = [Asking[i] for i in range(len(E_test))]
Asking[i]
len(E_test)
len(Err_Storage)
