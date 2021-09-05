# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 09:48:23 2020

@author: Omid
"""

def Get_Res_Worker_Kfast(param_master_loop):
    pass
    
def Get_Res_Worker(param_master_loop):    
    import torch
    import time
    from worker import worker_new
    from time_Kfast import t
    Kfast, Avg, learning_rate, Fix_net_state_dict, worker_data_loader, testloader, net, net_woker, MaxItr, MaxTime, smooth, criterion, VGG, CUDAA = param_master_loop
    time_start = time.time()
    Itr = 0
    dict_param2 = dict(net.named_parameters())
    Err = []
    TTime = 0
    Time = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Accuracy_SGD = 0 
    while TTime < MaxTime:
        net.load_state_dict(Fix_net_state_dict)
        cnterr = 0
        with torch.no_grad():
            for data in testloader:
                images_test, labels_test = data[0].to(device), data[1].to(device)
                outputs_SGD = net(images_test)
                _, predicted_SGD = torch.max(outputs_SGD, 1)
                c_SGD = (predicted_SGD == labels_test).squeeze()
                Accuracy_SGD +=  1-int(torch.sum(c_SGD))/len(labels_test)
                cnterr += 1
                break
        if Itr % 10 == 0:   
            if Itr == 0:   
                Err.append(Accuracy_SGD/(cnterr))
                Time.append(TTime)
                Accuracy_SGD = 0
            else:
                Err.append(Accuracy_SGD/(cnterr*10))
                print(Itr, Accuracy_SGD/(cnterr*10))
                Time.append(TTime)
                Accuracy_SGD = 0
        T, Participated_Workers = t(Kfast, Avg)
        TTime = T + TTime
############################################################################################################
        if Itr < 5:
            lr = learning_rate
        else:
            lr = learning_rate * Kfast 
        if Itr %200 == 199:
            lr = learning_rate * Kfast * .9
# Get from Workers #########################################################################################
        net_named_parameters = []
        loss_workers = 0
        for i in Participated_Workers:
            loss_worker, temp, temp2 = worker_new(Fix_net_state_dict, net_woker, worker_data_loader[i-1], lr)
            net_named_parameters.append(temp2)
            loss_workers += loss_worker/len(Participated_Workers)
        print('loss_workers', loss_workers)
############################################################################################################          
        for i, j in enumerate(net_named_parameters,0):   
            for name, param in j:
                if i == 0:
                    Fix_net_state_dict[name].data = dict_param2[name].data * 0
                if name in dict_param2:
                    Fix_net_state_dict[name].data = (param.data)/len(Participated_Workers) + Fix_net_state_dict[name].data
        Itr += 1
    print('time elapsed', (time.time() - time_start)/60,' min')
    return Err, Time