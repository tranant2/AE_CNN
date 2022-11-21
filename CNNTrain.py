# Neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

cwd = "/mnt/ufs18/home-032/tranant2/Desktop/MachineLearning/TRACK/CNN9"
test = 0
log_interval = 2

def train_model(model,
                epochs,
                train_loader,
                optimizer,
                device='cpu'):
    
    train_losses = []
    train_counter = []
    
    #set network to training mode
    model.train()
    for epoch in range(epochs):
        #iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                voltages = data[0].to(device)
                distro_0 = data[1].to(device)
                distro_1 = data[2].to(device)
                distro_2 = data[3].to(device)
                distro_3 = data[4].to(device)
                distro_4 = data[5].to(device)
                output_0 = target[0].to(device)
                output_1 = target[1].to(device)
                output_2 = target[2].to(device)
                output_3 = target[3].to(device)
                #reset gradients
                optimizer.zero_grad()

                #evaluate network with data
                distro, losses = model(voltages=voltages,
                                       distro0=distro_0,
                                       distro1=distro_1,
                                       distro2=distro_2,
                                       distro3=distro_3,
                                       distro4=distro_4,
                                       output0=output_0,
                                       output1=output_1,
                                       output2=output_2,
                                       output3=output_3)
                assert distro[0].dtype is torch.float16
                assert distro[1].dtype is torch.float16
                assert distro[2].dtype is torch.float16
                assert distro[3].dtype is torch.float16
                assert distro[4].dtype is torch.float16

                #compute loss and derivative
                #loss0 = nn.MSELoss()(output_0.to(device), losses[0])
                #loss1 = nn.MSELoss()(output_1.to(device), losses[1])
                #loss2 = nn.MSELoss()(output_2.to(device), losses[2])
                #loss3 = nn.MSELoss()(output_3.to(device), losses[3])
                loss4 = nn.MSELoss()(distro_0.to(device), distro[0])
                loss5 = nn.MSELoss()(distro_1.to(device), distro[1])
                loss6 = nn.MSELoss()(distro_2.to(device), distro[2])
                loss7 = nn.MSELoss()(distro_3.to(device), distro[3])
                loss8 = nn.MSELoss()(distro_4.to(device), distro[4])
                loss = (loss4+loss5+loss6+loss7+loss8) #+(loss0+loss1+loss2+loss3)
                loss.backward()
                assert loss.dtype is torch.float32
            #step optimizer
            optimizer.step()
            
            #print out results and save to file
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(voltages),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
                print(f"loss0:{0},loss1:{0}, loss2:{0}, loss3:{0},\n distro0:{loss4}, distro1:{loss5}, distro2:{loss6}, distro3:{loss7}, distro4:{loss8}")
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*1028) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), cwd + f'/results/model_{test}.pth')
            torch.save(optimizer.state_dict(), cwd + f'/results/optimizer_{test}.pth')
        
    return train_losses, train_counter

def train_model2(model,
                epochs,
                train_loader,
                optimizer,
                device='cpu'):
    
    train_losses = []
    train_counter = []
    
    #set network to training mode
    model.train()
    for epoch in range(epochs):
        #iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                voltages = data[0].to(device)
                distro_0 = data[1].to(device)
                distro_1 = data[2].to(device)
                distro_2 = data[3].to(device)
                distro_3 = data[4].to(device)
                distro_4 = data[5].to(device)
                output_0 = target[0].to(device)
                output_1 = target[1].to(device)
                output_2 = target[2].to(device)
                output_3 = target[3].to(device)
                #reset gradients
                optimizer.zero_grad()

                #evaluate network with data
                distro, losses = model(voltages=voltages,
                                       distro0=distro_0,
                                       distro1=distro_1,
                                       distro2=distro_2,
                                       distro3=distro_3,
                                       distro4=distro_4,
                                       output0=output_0,
                                       output1=output_1,
                                       output2=output_2,
                                       output3=output_3)
                assert distro[0].dtype is torch.float16
                assert distro[1].dtype is torch.float16
                assert distro[2].dtype is torch.float16
                assert distro[3].dtype is torch.float16
                assert distro[4].dtype is torch.float16

                #compute loss and derivative
                loss0 = nn.L1Loss(reduction='mean')(output_0.to(device), losses[0])
                loss1 = nn.L1Loss(reduction='mean')(output_1.to(device), losses[1])
                loss2 = nn.L1Loss(reduction='mean')(output_2.to(device), losses[2])
                loss3 = nn.L1Loss(reduction='mean')(output_3.to(device), losses[3])
                loss = (loss0+loss1+loss2+loss3)
                loss.backward()
                assert loss.dtype is torch.float32
            #step optimizer
            optimizer.step()
            
            #print out results and save to file
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(voltages),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
                print(f"loss0:{loss0},loss1:{loss1}, loss2:{loss2}, loss3:{loss3}")
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*1028) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), cwd + f'/results/model_{test}.pth')
            torch.save(optimizer.state_dict(), cwd + f'/results/optimizer_{test}.pth')
        
    return train_losses, train_counter

def train_model3(model,
                epochs,
                train_loader,
                optimizer,
                device='cpu'):
    
    train_losses = []
    train_counter = []
    
    #set network to training mode
    model.train()
    for epoch in range(epochs):
        #iterate through data batches
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                voltages = data[0].to(device)
                distro_0 = data[1].to(device)
                distro_1 = data[2].to(device)
                distro_2 = data[3].to(device)
                distro_3 = data[4].to(device)
                distro_4 = data[5].to(device)
                output_0 = target[0].to(device)
                output_1 = target[1].to(device)
                output_2 = target[2].to(device)
                output_3 = target[3].to(device)
                #reset gradients
                optimizer.zero_grad()

                #evaluate network with data
                distro, losses = model(voltages=voltages,
                                       distro0=distro_0,
                                       distro1=distro_1,
                                       distro2=distro_2,
                                       distro3=distro_3,
                                       distro4=distro_4,
                                       output0=output_0,
                                       output1=output_1,
                                       output2=output_2,
                                       output3=output_3)
                assert distro[0].dtype is torch.float16
                assert distro[1].dtype is torch.float16
                assert distro[2].dtype is torch.float16
                assert distro[3].dtype is torch.float16
                assert distro[4].dtype is torch.float16

                #compute loss and derivative
                loss0 = nn.L1Loss(reduction='mean')(output_0.to(device), losses[0])
                loss1 = nn.L1Loss(reduction='mean')(output_1.to(device), losses[1])
                loss2 = nn.L1Loss(reduction='mean')(output_2.to(device), losses[2])
                loss3 = nn.L1Loss(reduction='mean')(output_3.to(device), losses[3])
                loss4 = nn.MSELoss()(distro_0.to(device), distro[0])
                loss5 = nn.MSELoss()(distro_1.to(device), distro[1])
                loss6 = nn.MSELoss()(distro_2.to(device), distro[2])
                loss7 = nn.MSELoss()(distro_3.to(device), distro[3])
                loss8 = nn.MSELoss()(distro_4.to(device), distro[4])
                loss = (loss4+loss5+loss6+loss7+loss8)+(loss0+loss1+loss2+loss3)
                loss.backward()
                assert loss.dtype is torch.float32
            #step optimizer
            optimizer.step()
            
            #print out results and save to file
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(voltages),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
                print(f"loss0:{loss0},loss1:{loss1}, loss2:{loss2}, loss3:{loss3},\n distro0:{loss4}, distro1:{loss5}, distro2:{loss6}, distro3:{loss7}, distro4:{loss8}")
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*1028) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), cwd + f'/results/model_{test}.pth')
            torch.save(optimizer.state_dict(), cwd + f'/results/optimizer_{test}.pth')
        
    return train_losses, train_counter