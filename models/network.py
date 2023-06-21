import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt



class TrajNet(nn.Module):
    def __init__(self):
        super(TrajNet, self).__init__()
        self.rnn = nn.LSTM(input_size=6, hidden_size=64, num_layers=2)
        self.fc = nn.Linear(64, 1200)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        output = self.fc(h[-1])
        return output


class StackingBlock(nn.Module):
    def __init__(self, input, hidden, output):
        super(StackingBlock, self).__init__()

        layer_info = torch.cat((input, hidden, output))
        self.model = self.__gen_model(layer_info)
        

    def __gen_model(self, layer_info):
        layers = []

        for i in range(len(layer_info)-1):

            if i == len(layer_info)-2:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                
            else:
                layers.append(nn.Linear(layer_info[i], layer_info[i+1]))
                # layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Tanh())
                # layers.append(nn.BatchNorm1d(layer_info[i+1]))

        return nn.Sequential(*layers)


    def forward(self, coords):
        u = self.model(coords)
        
        return u