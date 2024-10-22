# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:06:00 2024

@author: sbhowmi2
"""
import torch
import torch.nn as nn
#from Models import QConv2D, QConv2D_AE, QConv2D_MF, Q_linear
from Quantum_convolution.QNN import Quanv2D_multi_filter, amplitude_embedding, angle_embedding, test_quanv, DressedQuantumNet
from Quantum_convolution.VQCs import vqc_num_params_dict
import pennylane as qml
import math
import torch.nn as nn


class QCONV_011(nn.Module):
    def __init__(self):
        super(QCONV_011, self).__init__()
        
        # First QConv layer <-- ip: (n, 1, 28, 28)
        VQC_circuit, VQC_num_params = vqc_num_params_dict[0]
        self.qconv1 = Quanv2D_multi_filter(in_channels = 1, kernel_size=2, stride=2, embedding=angle_embedding, VQC_circuit = VQC_circuit, VQC_n_layers = 2, VQC_num_params = VQC_num_params, n_filters = 8)
        # --> op: (n, 8, 14, 14)
        
        # 2nd QConv
        VQC_circuit, VQC_num_params = vqc_num_params_dict[1]
        self.qconv2 = Quanv2D_multi_filter(in_channels = 8, kernel_size=2, stride=2, embedding=amplitude_embedding, VQC_circuit = VQC_circuit, VQC_n_layers = 2, VQC_num_params = VQC_num_params, n_filters = 8)
        # --> op: (n, 8, 7, 7)
        
        # 3rd Qconv
        VQC_circuit, VQC_num_params = vqc_num_params_dict[1]
        self.qconv3 = Quanv2D_multi_filter(in_channels = 8, kernel_size=2, stride=2, embedding=amplitude_embedding, VQC_circuit = VQC_circuit, VQC_n_layers = 4, VQC_num_params = VQC_num_params, n_filters = 4)
        # --> op: (n, 4, 3, 3)
        
        # reverse MERA
        self.dqn = DressedQuantumNet(input_shape= 4*3*3, n_op = 10)
        
    def forward(self, x):
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = self.qconv3(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.dqn(x)
        return x

#%%       
        















    

        
