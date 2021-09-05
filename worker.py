def worker_new(Fix_net_state_dict, net_woker, worker_data_loader, lr):

    import torch
    import torch.nn as nn
    
    train_loader = worker_data_loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_woker.load_state_dict(Fix_net_state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_woker.classifier.parameters(), lr = lr, momentum = .9)
    running_loss = 0.0
    train_running_correct = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net_woker(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return running_loss, net_woker.parameters(), net_woker.named_parameters()



