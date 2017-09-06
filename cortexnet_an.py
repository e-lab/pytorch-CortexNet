#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:42:38 2017

@author: lburzawa
"""

import numpy as np
import torch
import cv2
from torch import nn
import torch.nn.functional as f
import torch.optim as optim
from torch.autograd import Variable as V
from math import ceil
import time
import copy
from torchvision.utils import save_image
from myVideoFolder import myVideoFolder
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
KERNEL_STRIDE = 2
OUTPUT_ADJUST = KERNEL_SIZE - 2 * PADDING


class Model02(nn.Module):
    """
    Generate a constructor for model_02 type of network
    """

    def __init__(self, network_size: tuple, input_spatial_size: tuple) -> None:
        """
        Initialise Model02 constructor
        :param network_size: (n, h1, h2, ..., emb_size, nb_videos)
        :type network_size: tuple
        :param input_spatial_size: (height, width)
        :type input_spatial_size: tuple
        """
        super().__init__()
        self.hidden_layers = len(network_size) - 1

        print('\n{:-^80}'.format(' Building model '))
        print('Hidden layers:', self.hidden_layers)
        print('Net sizing:', network_size)
        print('Input spatial size: {} x {}'.format(network_size[0], input_spatial_size))

        # main auto-encoder blocks
        self.activation_size = [input_spatial_size]
        for layer in range(0, self.hidden_layers):
            # print some annotation when building model
            print('{:-<80}'.format('Layer ' + str(layer + 1) + ' '))
            print('Bottom size: {} x {}'.format(network_size[layer], self.activation_size[-1]))
            self.activation_size.append(tuple(ceil(s / 2) for s in self.activation_size[layer]))
            print('Top size: {} x {}'.format(network_size[layer + 1], self.activation_size[-1]))

            # init D (discriminative) blocks
            multiplier = layer and 2 or 1  # D_n, n > 1, has intra-layer feedback
            setattr(self, 'D_' + str(layer + 1), nn.Conv2d(
                in_channels=network_size[layer] * multiplier, out_channels=network_size[layer + 1],
                kernel_size=KERNEL_SIZE, stride=KERNEL_STRIDE, padding=PADDING
            ))
            setattr(self, 'BN_D_' + str(layer + 1), nn.BatchNorm2d(network_size[layer + 1]))

            # init G (generative) blocks
            if layer==self.hidden_layers-1:
                multiplier=1
            else:
                multiplier=2
            setattr(self, 'G_' + str(layer + 1), nn.ConvTranspose2d(
                in_channels=network_size[layer + 1]*multiplier, out_channels=network_size[layer],
                kernel_size=KERNEL_SIZE, stride=KERNEL_STRIDE, padding=PADDING
            ))
            setattr(self, 'BN_G_' + str(layer + 1), nn.BatchNorm2d(network_size[layer]))
        self.fc1 = nn.Linear(256 * 8 * 10, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x, state):
        activation_sizes = [x.size()]  # start from the input
        residuals = list()
        state = state or [None] * (self.hidden_layers - 1)
        for layer in range(0, self.hidden_layers):  # connect discriminative blocks
            if layer:  # concat the input with the state for D_n, n > 1
                # s = state[layer - 1] or V(x.data.clone().zero_())
                if state[layer - 1] is None:
                    s = V(x.data.clone().zero_())
                else:
                    s = state[layer - 1]
                x = torch.cat((x, s), 1)
            x = getattr(self, 'D_' + str(layer + 1))(x)
            x = getattr(self, 'BN_D_' + str(layer + 1))(x)
            x = f.relu(x)
            residuals.append(x)
            activation_sizes.append(x.size())  # cache output size for later retrieval
        y = x.view(-1, 256 * 8 * 10)
        y = f.relu(self.fc1(y))
        y = self.dropout(y)
        y = self.fc2(y)
        for layer in reversed(range(0, self.hidden_layers)):  # connect generative blocks
            if layer != (self.hidden_layers-1):
                x=torch.cat((x,residuals[layer]),1)
            x = getattr(self, 'G_' + str(layer + 1))(x, activation_sizes[layer])
            if layer:
                x = getattr(self, 'BN_G_' + str(layer + 1))(x)
                x = f.relu(x)
                state[layer - 1] = x

        return (x, y, state)


train = myVideoFolder('/media/HDD1/Datasets/KTH/processed/Train')
test = myVideoFolder('/media/HDD1/Datasets/KTH/processed/Test')
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, drop_last=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, drop_last=False, num_workers=3)

print('Define model')
model = Model02(network_size=(3, 32, 64, 128, 256), input_spatial_size=(120, 160))
model = model.cuda()
print('Create a MSE and NLL criterions')
mse = nn.MSELoss()
nll = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    params=model.parameters(),
    lr=0.001
)
max_epoch = 20  # number of epochs

model_dir='./model_skip1'
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)
of = open('./model_skip1/train.txt', 'w')
og = open('./model_skip1/test.txt', 'w')
print('Run for', max_epoch, 'epochs')
for epoch in range(0, max_epoch):
    start_time = time.time()
    loss = 0
    loss_mse = 0.0;
    loss_nll = 0.0
    loss_mse_epoch = 0.0;
    loss_nll_epoch = 0.0
    model=model.train()
    for i, data in enumerate(train_loader, 0):
        x, _ = data
        x = x.permute(1, 0, 2, 3, 4)
        x = x.cuda()
        state = None
        (x_hat, y_hat, state) = model(V(x[0]), state)
        for t in range(1, 31):
            #if i==21:
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = 0.001
            (x_hat, y_hat, state) = model(V(x[t]), state)
            loss_temp = mse(x_hat, V(x[t + 1]))
            loss += loss_temp
            loss_mse += loss_temp.data[0]
            loss_mse_epoch += loss_temp.data[0]
            x_hat = x_hat.detach()
            state_enc = state
            for j in range(len(state_enc)):
                state_enc[j] = state_enc[j].detach()
            input_enc=torch.cuda.FloatTensor(64,3,120,160).zero_()
            label=torch.cuda.LongTensor(64).zero_()
            for j in range(64):
                if torch.rand(1)[0] < 0.5:
                    input_enc[j]=x_hat.data[j]
                    label[j]=0
                else:
                    input_enc[j]=x[t+1,j]
                    label[j]=1
            input_enc=V(input_enc.cuda())
            label=V(label.cuda())
            (x_hat_garbage, y_hat, state_garbage) = model(input_enc, state_enc)
            loss_temp = nll(y_hat, label)
            loss += loss_temp
            loss_nll += loss_temp.data[0]
            loss_nll_epoch += loss_temp.data[0]
            if (t % 6 == 0):
                model.zero_grad()
                loss.backward()
                optimizer.step()
                for j in range(len(state)):
                    state[j] = state[j].detach()
                loss = 0.0
        if i % 30 == 0:
            print(' > Epoch {:2d} Batch {:2d} loss_mse {:.5f} loss_nll {:.3f}'.format(epoch + 1, i, loss_mse / 900.0,
                                                                                      loss_nll / 900.0))
            loss_mse = 0.0;
            loss_nll = 0.0
            print('Time elapsed is %f' % (time.time() - start_time))
            start_time = time.time()
    of.write('%d,%f,%f\n' % (
    epoch + 1, loss_mse_epoch / (len(train_loader) * 30.0), loss_nll_epoch / (len(train_loader) * 30.0)));
    of.flush()
    print('\n==================================================\n==================================================')
    print(' > Training: Epoch {:2d} loss_mse: {:.5f} loss_nll: {:.3f}'.format(epoch + 1, loss_mse_epoch / (
    len(train_loader) * 30.0), loss_nll_epoch / (len(train_loader) * 30.0)))
    print('==================================================\n==================================================')
    torch.save(model.state_dict(), './model_skip1/model_{:02d}.pt'.format(epoch + 1))

    loss_mse = 0.0;
    correct = 0
    vid_len_total = 0;
    disp_count = 0
    disp_frames = torch.cuda.FloatTensor(10, 3, 120, 160)
    model=model.eval()
    for i, data in enumerate(test_loader, 0):
        x, _ = data
        x = x.permute(1, 0, 2, 3, 4)
        vid_len_total += (x.size(0) - 2)
        x = x.cuda()
        state = None
        (x_hat, y_hat, state) = model(V(x[0]), state)
        for t in range(1, x.size()[0] - 1):
            (x_hat, y_hat, state) = model(V(x[t]), state)
            if (i % 120 == 0 and t == 15):
                disp_frames[disp_count] = x_hat.data[0]
                disp_count += 1
            loss_mse += mse(x_hat, V(x[t + 1])).data[0]
            x_hat = x_hat.detach()
            state_enc = state
            for j in range(len(state_enc)):
                state_enc[j] = state_enc[j].detach()
            if torch.rand(1)[0] < 0.5:
                (x_hat_garbage, y_hat, state_garbage) = model(x_hat, state_enc)
                label = torch.LongTensor([0]).cuda()
            else:
                (x_hat_garbage, y_hat, state_garbage) = model(V(x[t + 1]), state_enc)
                label = torch.LongTensor([1]).cuda()
            _, predicted = torch.max(y_hat.data, 1)
            correct += (predicted == label).sum()
            for j in range(len(state)):
                state[j] = state[j].detach()
    save_image(disp_frames, './model_skip1/image{:02d}.png'.format(epoch + 1))
    og.write('%d,%f,%f\n' % (epoch + 1, loss_mse / vid_len_total, correct / vid_len_total));
    og.flush()
    print('\n==================================================\n==================================================')
    print(' > Testing: Epoch {:2d} loss_mse {:.5f} accuracy {:.3f}'.format(epoch + 1, loss_mse / vid_len_total,
                                                                           correct / vid_len_total))
    print('==================================================\n==================================================\n')

of.close()
og.close()

