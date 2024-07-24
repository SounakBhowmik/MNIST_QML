# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:06:00 2024

@author: sbhowmi2
"""
import torch
import torch.nn as nn
from Models import QConv2D, QConv2D_AE, QConv2D_MF, Q_linear


class QNNModel_1(nn.Module):
    
    '''
        Used Angle embedding; Strongly entangling layers; 1 Qc layer (9 qubbits, 3x3 filter, strongly entangled circuit) + dense layer at the end.
    
    '''
    def __init__(self, device=None):
        super(QNNModel_1, self).__init__()
        self.qconv1 = QConv2D(in_channels=1, kernel_size=3, n_layers=1, stride=2, device=device) # 9, 13, 13
        
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)
        
        self.fc1 = nn.Linear(9 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        
    def forward(self,x):
        x = self.pool(self.qconv1(x)) # 9, 5, 5
	
        x = torch.relu(x)
       
        x = x.view(-1, 9 * 5 * 5)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        
        return x
        
