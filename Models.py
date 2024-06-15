# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import pennylane as qml
from torch.nn import Module
from pennylane import numpy as np
import math

# Define the CNN model for binary classification with kernel size 8
class ClassicalModel1(Module):
    def __init__(self):
        super(ClassicalModel1, self).__init__()
        self.name = "ClassicalModel1"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride = 2) # -1, 32, 99, 99
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2) # -1, 32, 48, 48

        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride = 2) # -1, 64, 21, 21
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)     # -1, 64, 10, 10

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride = 2)            # -1, 128, 4, 4

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x




class ClassicalModel2(Module):
    def __init__(self):
        super(ClassicalModel2, self).__init__()
        self.name = "ClassicalModel2"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride = 2) # -1, 32, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2) # -1, 32, 47, 47

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride = 2) # -1, 64, 22, 22
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)     # -1, 64, 19, 19

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride = 1)            # -1, 128, 18, 18
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=2)     # -1, 128, 8, 8

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x
    
    
class ClassicalModel3(Module):
    def __init__(self):
        super(ClassicalModel3, self).__init__()
        self.name = "ClassicalModel3"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=16, stride = 2) # -1, 32, 93, 93
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2) # -1, 32, 45, 45

        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride = 2) # -1, 64, 19, 19
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)     # -1, 64, 18, 18

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride = 1)            # -1, 128, 17, 17
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)     # -1, 128, 16, 16

        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2) # -1, 128, 7, 7
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)     # -1, 128, 3, 3

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))
        x = self.maxpool4(torch.relu(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x
    
class ClassicalModel4(Module):
    def __init__(self):
        super(ClassicalModel4, self).__init__()
        self.name = "ClassicalModel4"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 32, stride = 2) # -1, 32, 85, 85
        self.maxpool1 = nn.MaxPool2d(kernel_size = 8, stride = 1) # -1, 32, 78, 78

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 16, stride = 2) # -1, 64, 32, 32
        self.maxpool2 = nn.MaxPool2d(kernel_size=8, stride=1)     # -1, 64, 25, 25

        self.conv3 = nn.Conv2d(64, 128, kernel_size = 2, stride = 1)            # -1, 128, 24, 24
        self.maxpool3 = nn.MaxPool2d(kernel_size = 8, stride = 1)     # -1, 128, 17, 17

        self.conv4 = nn.Conv2d(128, 128, kernel_size = 2, stride = 1)            # -1, 128, 16, 16
        self.maxpool4 = nn.MaxPool2d(kernel_size = 8, stride = 2)     # -1, 128, 5, 5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 5 * 5, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(p=0.12)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))
        x = self.maxpool4(torch.relu(self.conv4(x)))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.dropout3(torch.relu(self.fc3(x)))

        x = torch.relu(self.fc4(x))

        x = torch.log_softmax(self.fc5(x), dim=1)
        return x
    
# Hybrid QNN ####################################################################################################
class Q_linear(Module):
    in_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features: int, n_layers:int, 
                 bias: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.dev = qml.device("lightning.gpu", wires = in_features) if torch.cuda.is_available() else qml.device("default.qubit", wires = in_features)

        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            #print(f"#################weights = {weights}#################")
            '''
            # Hadamard Layer
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.in_features))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.in_features))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.in_features)]
        
        weight_shapes = {"weights": (self.n_layers, self.in_features, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes= weight_shapes)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.qlayer(input)

### This is a class for using Quanvolution with any torch model
class Q_conv(Module):
    def __init__(self,  kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(Q_conv, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(self.kernel_size**2)
        self.n_layers = n_layers
        self.stride = stride
        
        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        #init_method = {"weights": self.weights}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        #dev = qml.device("default.qubit", wires = n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            #print(f"#################weights = {weights}#################")
            '''
            # Hadamard Layer # Increases complexity and time of training
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        #self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output


class HybridModel1(Module):
    def __init__(self):
        super(HybridModel1, self).__init__()
        self.name = "HybridModel1"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride = 2) # -1, 32, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2) # -1, 32, 47, 47

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride = 2) # -1, 64, 22, 22
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)     # -1, 64, 19, 19

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride = 1)            # -1, 128, 18, 18
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=2)     # -1, 128, 8, 8

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 5)
        self.qlayer = Q_linear(5, 3)
        self.fc4 = nn.Linear(5,2)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.sigmoid(self.fc3(x))*np.pi
        x = self.qlayer(x)
        x = torch.log_softmax(self.fc4(x), dim=1)
        return x
    
class Hybrid_QuanvModel(Module):
    def __init__(self):
        super(Hybrid_QuanvModel, self).__init__()
        self.name = "Hybrid_QuanvModel"
        # Input shape: -1, 1, 200, 200
        self.conv1 = nn.Conv2d(1, 1, kernel_size=8, stride = 2) # -1, 32, 97, 97
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2) # -1, 1, 47, 47
        
        self.q_conv = Q_conv(2, 3, stride=2)                    # -1, 4, 23,23 
        
        self.conv2 = nn.Conv2d(4, 32, kernel_size=4, stride = 1) # -1, 32, 20, 20
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)    # -1, 32, 17, 17

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride = 1) # -1, 64, 14, 14
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)     # -1, 64, 13, 13
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride = 2) # -1, 128, 5, 5

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 5 * 5, 128)  # Adjusted based on the new feature dimensions
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 64)  # Output 2 value for binary classification
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,2)

    def forward(self, x):
        x = self.maxpool1(torch.tanh(self.conv1(x)))*np.pi/2
        x = torch.relu(self.q_conv(x))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = self.maxpool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        
        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))

        x = torch.relu(self.fc3(x))

        x = torch.log_softmax(self.fc4(x), dim=1)
        return x
'''
class DressedQuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # The output from the penultimate layer of the classical models has 16 dimensions
        self.pre_net = nn.Linear(16, 5)
        self.qlayer = Q_linear(in_features=5, n_layers=3)
        self.post_net = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.tanh(self.pre_net(x)) * np.pi / 2.0
        x = torch.relu(self.qlayer(x))
        x = self.post_net(x)
        return x
'''   
class DressedQuantumNet(nn.Module):
    def __init__(self, input_shape, n_qubits=5, n_layers=3, n_op = 2):
        super().__init__()
        # The output from the penultimate layer of the classical models has 16 dimensions
        self.pre_net = nn.Linear(input_shape, n_qubits)
        self.qlayer = Q_linear(in_features=n_qubits, n_layers=n_layers)
        self.post_net = nn.Linear(n_qubits, n_op)

    def forward(self, x):
        x = torch.tanh(self.pre_net(x)) * torch.pi / 2.0
        x = torch.relu(self.qlayer(x))
        x = torch.log_softmax(self.post_net(x), dim=1)
        return x
    
    
class DressedClassicalNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # The output from the penultimate layer of the classical models has 16 dimensions
        self.fc1 = nn.Linear(input_shape, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x
#%%
'''
import time
hqnn = Hybrid_QuanvModel()
x = torch.rand(1000, 1, 200, 200)
start = time.time()
y = hqnn(x)
print(time.time()-start)
'''
#%% Full-fledged quanvolution
class QConv2D(nn.Module):
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(QConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(in_channels * self.kernel_size**2)
        self.n_layers = n_layers
        self.stride = stride

        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        qnode = qml.QNode(self.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def quantum_circuit(self, inputs, weights):
            '''
            # Hadamard Layer # Increases complexity and time of training
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        #assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output

#%% Full-fledged quanvolution with amplitude encoding
class QConv2D_AE(nn.Module):
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(QConv2D_AE, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(math.ceil(math.log(in_channels * self.kernel_size**2, 2)))
        self.n_layers = n_layers
        self.stride = stride

        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        qnode = qml.QNode(self.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def quantum_circuit(self, inputs, weights):
        '''
        # Hadamard Layer # Increases complexity and time of training
        for wire in range(n_qubits):
            qml.Hadamard(wires = wire)
        '''
        # Embedding layer
        qml.AmplitudeEmbedding(features = inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.)
        # Variational layer
        for _ in range(self.n_layers):
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        #assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output







