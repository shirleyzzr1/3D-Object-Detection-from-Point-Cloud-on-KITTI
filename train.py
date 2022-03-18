from sklearn.utils import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import KittiDataset
from model import PointNet
import torch.optim as optim
import torch.nn as nn
import numpy as np
def make_weights_for_balanced_classes(data, num):

    N = float(sum(num))   
    weights = [0.] * len(data)   
    weights[:num[0]]=  (N/num[0])*np.ones(num[0])          
    for i in range(1,len(num)):
        weights[num[i-1]:num[i-1]+num[i]]= (N/num[i])*np.ones(num[i])                                              
    return weights  
if __name__=="__main__":
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = KittiDataset("./data/",split="training")
    test_data = KittiDataset("./data/",split="val")
    weights = make_weights_for_balanced_classes(train_data.filenames, train_data.num)
    weights = torch.FloatTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    # print(weights)
    writer = SummaryWriter('./output/runs/tensorboard')
    batch_size = 64
    best_acc = 0
    trainloader = torch.utils.data.DataLoader(train_data,batch_size = batch_size,shuffle=True,drop_last=False)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = batch_size,shuffle=True,drop_last=False)
    print("device",device)
    net = PointNet()
    decay_lr_factor = 0.95
    decay_lr_every = 2
    lr = 0.01
    #define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
          optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    for epoch in range(5):  # loop over the dataset multiple times
        training_correct = 0
        training_loss = 0.0
        for i, data in enumerate(trainloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            # inputs = torch.from_numpy(inputs.numpy())
            inputs = inputs.to(torch.float32)
            #net = torch.nn.DataParallel(net)
            net.to(device)
            # net = torch.nn.DataParallel(net)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            training_correct+=correct
            # if i % 20 == 1:    # print every 20 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            #     running_loss = 0.0

            # print('Finished Training')
            if i%5==0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() /batch_size:.3f}')
            print("epoch",epoch,"i",i)
            
        print('training acc', training_correct / len(train_data))
        writer.add_scalar('training loss', training_loss / len(train_data),epoch)
        writer.add_scalar('training acc', training_correct / len(train_data),epoch)
        #update the learning rate
        scheduler.step()
        testing_correct = 0
        testing_loss = 0.0
        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device),labels.to(device)
            net.eval()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            testing_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  
            correct = (predicted == labels).sum().item()
            testing_correct+=correct
            print("epoch",epoch,"i",i)
        correctness = testing_correct / len(test_data)
        writer.add_scalar('test loss', testing_loss / len(test_data),epoch)
        writer.add_scalar('test acc', correctness,epoch)
        print('test acc',correctness)

        if correctness>best_acc:
            best_acc = correctness
            torch.save(net.state_dict(), './model/model_%d.pth'%(epoch))


    