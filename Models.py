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


# Hybrid QNN ####################################################################################################
class Q_linear(Module):
    in_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features: int, n_layers:int, 
                 bias: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.dev = (qml.device("lightning.gpu", wires = in_features) if torch.cuda.is_available() else qml.device("default.qubit", wires = in_features)) if device == None else device

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

#%% VQC designs ####################################################################################################

def angleEmbedding(n_qubits: int, inputs: torch.Tensor):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

def quantum_circuit_default(n_qubits: int, params: torch.Tensor, n_layers: int):
    # Variational layer
    for _ in range(n_layers):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))

def quantum_circuit_1(n_qubits: int, params: torch.Tensor, n_layers: int): #2, 4, 3
    for l in range(n_layers):
        # Apply RX and RZ gates to each qubit
        for i in range(n_qubits):
            qml.RX(params[l][i, 0], wires=i)
            qml.RZ(params[l][i, 1], wires=i)
        
        # Apply CNOT gates
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i+1, i])

def quantum_circuit_16(n_qubits: int, params: torch.Tensor, n_layers: int):
    for l in range(n_layers):
        # Apply RX and RZ gates to each qubit
        for i in range(n_qubits):
            qml.RX(params[l][i, 0], wires=i)
            qml.RZ(params[l][i, 1], wires=i)
        
        # Apply CRZ gates
        for i in range(n_qubits-1):
            qml.CRZ(phi=params[l][i, 2], wires = [i+1, i])

def quantum_circuit_14(n_qubits: int, params: torch.Tensor, n_layers: int):
    for l in range(n_layers):
        # Apply Ry gates
        for i in range(n_qubits):
            qml.RY(params[l][i, 0], wires=i)
        
        # Apply CRX gates
        for i in range(n_qubits):
            w = (i+1) if (i+1<n_qubits) else i+1-n_qubits
            qml.CRX(phi=params[l][i, 1], wires=[i, w])
        
        # Apply Ry gates
        for l in range(n_layers):
            # Apply Ry gates
            for i in range(n_qubits):
                qml.RY(params[l][i, 2], wires=i)
        
        # Apply CRX gates
        for i in range(n_qubits):
            w = (i+1) if (i+1<n_qubits) else i+1-n_qubits
            qml.CRX(phi=params[l][i, 1], wires=[w, i])
            

def quantum_circuit_11(n_qubits: int, params: torch.Tensor, n_layers: int):
    for l in range(n_layers):
        # Apply RY and RZ gates
        for i in range(n_qubits):
            qml.RY(params[l][i, 0], wires=i)  
            qml.RZ(params[l][i, 1], wires=i)
        # Apply C-NOTs
            for i in range(0, n_qubits-1, 2):
                qml.CONT(wires=[i+1, i])
        
        # Apply RY and RZ gates
        for i in range(1, n_qubits-1):
            qml.RY(params[l][i, 2], wires=i)  
            qml.RZ(params[l][i, 3], wires=i)
        # Apply C-NOTs
            for i in range(1, n_qubits-1, 2):
                qml.CONT(wires=[i+1, i])
        

            


QUANTUM_CIRCUITS_DICT = [{"id":0, "circuit":quantum_circuit_default, "weight_shape":[None, None, 3]}, 
                         {"id":1, "circuit":quantum_circuit_1, "weight_shape":[None, None, 2]},
                         {"id":16, "circuit":quantum_circuit_16, "weight_shape":[None, None, 3]},
                         {"id":14, "circuit":quantum_circuit_14, "weight_shape":[None, None, 4]},
                         {"id":11, "circuit":quantum_circuit_11, "weight_shape":[None, None, 4]}]

##################################################################################################################
#%% Full-fledged quanvolution
class QConv2D(nn.Module):
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, ckt_id=1, dtype=None)->None:
        super(QConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(in_channels * self.kernel_size**2)
        self.n_layers = n_layers
        self.stride = stride
        
        # First define a q-node
        self.weight_shape = None
        self.vqc = None
        for item in QUANTUM_CIRCUITS_DICT:
            if(item["id"] == ckt_id):
                self.vqc = item["circuit"]
                self.weight_shape = item["weight_shape"]
                break
                
        self.weight_shape[0] = self.n_layers
        self.weight_shape[1] = self.n_qubits
        
        self.params = torch.nn.Parameter(torch.tensor(np.random.random(tuple(self.weight_shape)), requires_grad=True))
        
        weight_shapes = {"weights": self.weight_shape}
        dev = device if (device is not None) else qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        qnode = qml.QNode(self.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    def quantum_circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        angleEmbedding(self.n_qubits, inputs)
        self.vqc(self.n_qubits, self.params, self.n_layers)
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
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, dtype=None, vqc=0)->None:
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


#%% Full-fledged quanvolution with multiple filters
n_qubits = None
q_device = None


class QConv2D_MF(nn.Module):
    '''
        It will be able to use multiple VQCs, aka multiple filters
    '''
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, filters: int, device=None, dtype=None)->None:
        super(QConv2D_MF, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(math.ceil(math.log(in_channels * self.kernel_size**2, 2)))
        self.n_layers = n_layers
        self.stride = stride
        self.filters = filters

        self.q_device = (qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)) if device == None else device
        
        # Initialize the parameters
        weight_shape = (self.filters, self.n_layers, self.n_qubits, 3) 
        self.weights = nn.Parameter(torch.tensor(np.random.randn(*weight_shape), dtype=torch.float32))
        
        self.qfilters = []
        for i in range(self.filters):
            qnode = qml.QNode(self.quantum_circuit, self.q_device, interface='torch')
            self.qfilters.append(qml.qnn.TorchLayer(qnode, {'weights': self.weights[i,:].shape}))
            
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
    
    def forward_singlefilter(self, q_filter: qml.qnn.TorchLayer, input: torch.Tensor) -> torch.Tensor:

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
                exp_vals = q_filter(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = None
        for i in range(self.filters):
            if output is not None:
                output = torch.concat((output, self.forward_singlefilter(self.qfilters[i], input)), dim=1)
            else:
                output = self.forward_singlefilter(self.qfilters[i], input)
        return output


