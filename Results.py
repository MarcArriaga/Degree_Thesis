import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
from os import walk
import os

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision
import torch.nn as nn

import torchvision.models as models

from torch.autograd import Variable

import numpy as np

from torch.utils.tensorboard import SummaryWriter

path_train = "/mnt/gpid07/imatge/marc.arriaga/database/paths/splits/train_2_rgb.txt"
path_test =  "/mnt/gpid07/imatge/marc.arriaga/database/paths/splits/test_2_rgb.txt"

'''
This code contains the training and validation process of the analysis of the results. 
It also inculdes the visualization of all these results. 

This code uses:
- Two Resnet 50 Model to predict the Yaw and Pitch angles each one.
- Custom loss function
- Epochs: 160
- Learning Rate: 0.001
- Optimizer: SGD


Tensorboard displays the yaw and pitch loss for all the epochs,
as well as the positions x,y in all the frames.
It also shows a comparison between the prediced yaw and pitch angle
and the ground truth for each trajectory.
'''

writer = SummaryWriter('runs/Final/Results2')


def angular_error(a1,a2):
    #Custom loss function
    phi = torch.abs(a1 - a2) % 360
    phi = torch.mean(phi)
    dist = 360 - phi if phi > 180 else phi
    return dist


class AngleData(Dataset):
    def __init__(self, split, transform=None):
        self.paths = pd.read_csv(split, sep=' ', header=None)[0].values.tolist()
        self.transform = transform
        self.samples = []
        self.posx = pd.read_csv(split, sep=' ', header=None)[4].values.tolist()
        self.posy = pd.read_csv(split, sep=' ', header=None)[5].values.tolist()
        self.labels = [pd.read_csv(split, sep=' ', header=None)[1].values.tolist(), pd.read_csv(split, sep=' ',header=None)[2].values.tolist()]     
        self.mean = 0
        self.var = 0
        for i in range(len(self.paths)):
            self.samples.append((self.paths[i], [self.labels[0][i], self.labels[1][i]],
                [self.posx[i][1:self.posx[i].find(',')], self.posy[i][:-1]]))
            image = Image.open(self.paths[i])
            pixels = np.asarray(image)
            pixels = pixels.astype('float32')
            self.mean += pixels.mean()
            self.var += pixels.var()

        self.mean /= len(self.paths)
        self.var /= len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        path, label, pos = self.samples[idx]
        img = Image.open(path)


        img_arr = np.asarray(img)

        img_arr = (img_arr - self.mean) / self.var 

        if self.transform:
            img = self.transform(img)

        return img, label, pos

train_dataset = AngleData(split= path_train,
        transform = transforms.Compose([
            transforms.ToTensor()
            ]))
test_dataset = AngleData(split=path_test,
        transform = transforms.Compose([
            transforms.ToTensor()
            ]))


train_loader = DataLoader(dataset=train_dataset,
        batch_size=4,
        shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
        batch_size=4,
        shuffle=False)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1))

    def forward(self, x):
        out1 = self.model_resnet(x)
        return out1

model_yaw = MyModel()
model_pitch = MyModel()

def initModel(model, use_pretrained=True):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

model_yaw = model_yaw.cuda()
model_pitch = model_pitch.cuda()

def train(epoch, model_yaw, model_pitch, optimizer_yaw, optimizer_pitch, criterion, loader):
    model_yaw.train()
    model_pitch.train()
    loss_h_tot = 0
    loss_v_tot = 0
    i = 0
    for data in loader:
        i += 1
        inputs, labels, pos = data
        inputs, labels = inputs.cuda(), [labels[0].cuda(), labels[1].cuda()]
        
        optimizer_yaw.zero_grad()
        optimizer_pitch.zero_grad()

        outputsH = model_yaw(inputs.float()) 
        outputsV = model_pitch(inputs.float())

        outputsH = outputsH.double()
        outputsV = outputsV.double()
        outputsH = outputsH.squeeze(-1)
        outputsV = outputsV.squeeze(-1)

        loss_h = angular_error(labels[0], outputsH)
        loss_v = angular_error(labels[1], outputsV)
        loss_h_tot += loss_h.item()
        loss_v_tot += loss_v.item()
        loss_h.backward()
        loss_v.backward()
        
        optimizer_yaw.step()
        optimizer_pitch.step()
    writer.add_scalar('Loss/train_H', loss_h_tot/i, epoch)
    writer.add_scalar('Loss/train_V', loss_v_tot/i, epoch)
    print('Train Epoch: {} Loss: Horizontal {:.6f} Vertical: {:.6f}'.format(epoch+1, loss_h_tot/(i), loss_v_tot/(i)))
    return loss_h_tot/i, loss_v_tot/i

def test(model_yaw, model_pitch, criterion, loader):
    
    model_yaw.eval()
    model_pitch.eval()
    i = 0
    loss_h_tot = 0
    loss_v_tot = 0
    for data in loader:
        i += 1
        inputs, labels, pos = data
        inputs, labels = inputs.cuda(), [labels[0].cuda(), labels[1].cuda()]

    
        outputsH = model_yaw(inputs.float()) 
        outputsV = model_pitch(inputs.float())
        
        outputsH = outputsH.double()
        outputsV = outputsV.double()

        outputsH = outputsH.squeeze(-1)
        outputsV = outputsV.squeeze(-1)

        loss_h = angular_error(labels[0], outputsH) 
        loss_v = angular_error(labels[1], outputsV)
        loss_h_tot += loss_h.item()
        loss_v_tot += loss_v.item()
    writer.add_scalar('Loss/test_H', loss_h_tot/i, epoch)
    writer.add_scalar('Loss/test_V', loss_v_tot/i, epoch)
    print('\nTest set: Average loss: Horizontal {:.4f} Vertical {:.4f}\n'.format(loss_h_tot/(i), loss_v_tot/(i)))
    return loss_h_tot/i, loss_v_tot/i


criterion = nn.MSELoss()
optimizer_yaw = torch.optim.SGD(model_yaw.parameters(), lr=0.001, momentum=0.9)
optimizer_pitch = torch.optim.SGD(model_pitch.parameters(), lr=0.001, momentum=0.9)

train_losses = []
test_losses = []
for epoch in range(160):
    train(epoch, model_yaw, model_pitch, optimizer_yaw, optimizer_pitch, criterion, train_loader)
    test(model_yaw, model_pitch, criterion, test_loader)

print('Finished Training')

root = '/mnt/gpid07/imatge/marc.arriaga/database/paths'

files = [w for w in os.listdir(root) if w != 'split.py' or w != 'splits']

for file1 in files:
    if file1 != 'split.py' and file1 != 'splits':
        val_dataset = AngleData(split=os.path.join(root,file1),
                transform = transforms.Compose([
                    transforms.ToTensor()
                    ]))
        val_loader = DataLoader(dataset=val_dataset,
                batch_size=4,
                shuffle=False)

        labels_yaw = []
        labels_pitch = []
        pred_yaw = []
        pred_pitch = []
        position_x = []
        position_y = []


        for data in val_loader:

            images, labels, pos = data
            inputs = images.cuda()


            inputs = inputs.float()
            outputH = model_yaw(inputs) 
            outputV = model_pitch(inputs)

            for i in range(len(labels[0])):
            
                labels_yaw.append(labels[0][i].item())
                labels_pitch.append(labels[1][i].item())
                pred_yaw.append(outputH[i].item())
                position_x.append(float(pos[0][i]))
                position_y.append(float(pos[1][i]))
                pred_pitch.append(outputV[i].item())
                print('Yaw {} / {} \n Pitch {} / {} \n'.format(labels[0][i].item(), outputH[i].item(), labels[1][i].item(), outputV[i].item()))
        
        for j in range(len(labels_yaw)):
            writer.add_scalars('{} Yaw: '.format(file1), {'Pred':pred_yaw[j],
                                        'truth':labels_yaw[j]}, j)
            writer.add_scalars('{} Pitch: '.format(file1), {'Pred':pred_pitch[j],
                                        'truth':labels_pitch[j]}, j)
            writer.add_scalars('{} Position: '.format(file1), {'x':position_x[j],
                                            'y':position_y[j]}, j)
