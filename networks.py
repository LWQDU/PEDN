#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
from unet import MSDFN

class IRN(nn.Module):
    def __init__(self, recurrent_iter, use_GPU=True):
        super(IRN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU
        self.e_d = MSDFN()
        self.conv0 = nn.Sequential(#此处有更改，迭代版变回6输入
            nn.Conv2d(6, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)   
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            #nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            )
 
    def forward(self, input, depth):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input

        h = Variable(torch.zeros(batch_size, 64, row, col))
        c = Variable(torch.zeros(batch_size, 64, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = x
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            #######################
            """conv"""
            x = self.conv0(x)
            #LSTM
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            ##########################

            x = h
            #####################################################
            x = self.e_d(x, depth)
            ########################################################

            #########################################################
            """conv"""
            #x = self.conv(x)

            #x = x + input
            ########################################################
            x_list = torch.cat((x_list, x),1)
        return x,x_list
