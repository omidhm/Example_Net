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


learning_rate = 0.001
untrained_model_name= 'untrained_model.pkl'
trained_model_name= 'trained_model.pkl'
merged_model_name= 'merged_model.pkl'
ModelList = []          ## list for models
Res = []
MaxItr = 1000
batch_size = 3
smooth = 100


n_workers = 1
Kfast = 1
Avg = n_workers

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
                              
#Split the test sets                            
testSize = 1000                                      
TestSet1, TestSet2 = torch.utils.data.random_split(testset, [testSize,(len(testset)-testSize)])


temp = tuple([len(testset)//n_workers for i in range(n_workers)])
dataset = torch.utils.data.random_split(testset, temp)


worker_data_loader = []
for data_w in dataset:
    worker_data_loader.append(torch.utils.data.DataLoader(data_w, batch_size=batch_size, shuffle=True, num_workers=0))
             
                                       
testloader = torch.utils.data.DataLoader(TestSet1, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    
    
    
    
time_start = time.time()
#Make fix model for start

Fix_net = model()
Fix_net_par1 = Fix_net.named_parameters()
Fix_net_par = copy.deepcopy(dict(Fix_net_par1))

# Start
#upload the net parameters to the net()
net = model()
#$$$$$$$$$$$$$$$$$$$$$
#net.load_state_dict(Fix_net_par)
#Intital_worker_state = net.state_dict()
#$$$$$$$$$$$$$$$$$$$$$
####301020####
Itr = 0
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

loss_workers = 0
Err = []
TTime = 0
Time = []
while Itr < MaxItr:
    net_parameters = []
    optimizer.zero_grad()
    net.load_state_dict(Fix_net_par)
    
    # Timing for Kfast:
    T, Participated_Workers = t(Kfast, Avg)
    TTime = T + TTime
    
    
    
    for i in Participated_Workers:
        # 1. worker_data_loader send once to the worker(prevent network ...)
        # 2. Initial_worker_state sends to worker in each iteration.
        #OutPut:
        # 1. loss_worker
        # 2. temp : net_parameters
        loss_worker, temp = worker_new(Fix_net_par, worker_data_loader[i])
        net_parameters.append(temp) 
        loss_workers += loss_worker/len(Participated_Workers)
        
    # Updating model
        
    for P in zip(net.parameters(), *net_parameters):
#        avg = (PA.grad + PB.grad+PC.grad + PD.grad)/4
#        avg = (PA.grad + PB.grad)/2
        #        avg = PA.grad
#        a = [p.grad for p in P[1:]]
#        torch.sum(torch.Tensor(a))
#        np.sum(a)
#        a[0] + a[1]
        
        ### Short form
        temp = 0
        for p in P[1:]:
            temp += p.grad/len(Participated_Workers)
#            print(temp)
        P[0].grad = temp
        ### Short form
        
#        for i in zip(*a):
#            print(torch.add(i))
#            print(torch.add(a[i]))
#            
#        torch.add([p.grad for p in P[1:]])
#        torch.add(a)
#        P[0].grad = np.average([p.grad for p in P])
#         PA.grad = avg.clone()
        # PB.grad = avg.clone()
#        P.grad = avg.clone()
    
#    for P in net.parameters():
#        print(P.grad)
    
    optimizer.step() 
    
    # ? nemidunam lazeme ya na!?
    for name, param in net.named_parameters():
            Fix_net_par[name].data.copy_(param.data)
    
    if Itr == 0:
        Time.append(TTime)
        Err.append(loss_workers)
    
    if Itr %smooth == smooth-1:
        Time.append(TTime)
        Err.append(loss_workers / smooth)
        loss_workers = 0
        print(Itr)
    
 
    Itr += 1


    
# Test th net:
criterion = nn.CrossEntropyLoss()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Number of workers:%d Accuracy of %5s : %2d %%' % (Kfast, classes[i], 100 * class_correct[i] / class_total[i]))

print('time elapsed', (time.time() - time_start)/60,' min')

# return (Err, Time)



#plt.plot(Err10)
for i in range(len(Err)-1):
#    print(i)
    plt.plot(Time[i],Err[i],'bo',(Time[i],Time[i+1]), (Err[i],Err[i+1]), 'b--')
    



#Err2 = Err

####################


     
#Grad = [[] for i in range(n_workers)]
#for i in range(n_workers):
#    print('hi')
#    parameters_w = worker_new(Intital_worker_state, i) 
#    
#
#    for p in parameters_w:
#        Grad[i].append(p.grad)
#     
#G = []
#for j in range(10):
#    G.append( Grad[0][j]/2 + Grad[1][j]/2 )
#     
#
#cnt_up = 0
#for p in net.parameters():
#    print(p.grad)
#    p.grad = torch.Tensor(G[cnt_up])
#    cnt_up += 1
#
#      
#
#
#    # print(type(Grad))
#    
#    # tempModel = model()
#    
#    # tempModel.load_state_dict(Res[i])
#    # ModelList.append(tempModel)




########################################################


#
#MergedModel = model()
#MergedModel.load_state_dict( ModelList[1].state_dict())
#
#criterion = nn.CrossEntropyLoss()
#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))
#
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        outputs = MergedModel(images)
#        loss = criterion(outputs, labels)
#        
#        _, predicted = torch.max(outputs, 1)
#        c = (predicted == labels).squeeze()
#        for i in range(4):
#            label = labels[i]
#            class_correct[label] += c[i].item()
#            class_total[label] += 1
#
#
#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))
#
#
#
#
###########################
#
#
#MergedModel = model()
#MergedModel.load_state_dict( ModelList[0].state_dict())
#
#criterion = nn.CrossEntropyLoss()
#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))
#
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        outputs = MergedModel(images)
#        loss = criterion(outputs, labels)
#        
#        _, predicted = torch.max(outputs, 1)
#        c = (predicted == labels).squeeze()
#        for i in range(4):
#            label = labels[i]
#            class_correct[label] += c[i].item()
#            class_total[label] += 1
#
#
#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))
#
#
#########################################################
#
#beta = 0.5 #
#print("Reading models")
#params1 = ModelList[0].named_parameters()
#params2 = ModelList[1].named_parameters()
#
#dict_params2 = dict(params2)
#
#print("Merging models")
#for name1, param1 in params1:
#    if name1 in dict_params2:
#        dict_params2[name1].data.copy_(beta*param1.data + (1-beta)*dict_params2[name1].data)
#
#print("Start testing the merged model")
#MergedModel = model()
#MergedModel.load_state_dict(dict_params2)
#
#criterion = nn.CrossEntropyLoss()
#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))
#
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        outputs = MergedModel(images)
#        loss = criterion(outputs, labels)
#        
#        _, predicted = torch.max(outputs, 1)
#        c = (predicted == labels).squeeze()
#        for i in range(4):
#            label = labels[i]
#            class_correct[label] += c[i].item()
#            class_total[label] += 1
#
#
#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))


