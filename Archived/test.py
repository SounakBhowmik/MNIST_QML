import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
 
# Define the quantum device
dev = qml.device('ionq.simulator', wires=2, shots=1024, api_key='z91NPV3A3PEc7zE1Uh9Vaw4Q4DNFAJoR')
#dev = qml.device('default.qubit')
 
# You can also use the simulator for testing
# dev = qml.device('ionq.simulator', wires=2, shots=1024, api_key='YOUR_IONQ_API_KEY')
 
# Define the quantum circuit
@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=[0, 1])
    qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))  # Return only one expectation value
    #return [qml.expval(qml.PauliZ(wires=i)) for i in range(2)]
 
# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(HybridModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        # Quantum layer
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)
        # Classical layers
        self.clayer_1 = nn.Linear(1, 4)  # Adjust input size
        self.clayer_2 = nn.Linear(4, 1)
    def forward(self, x):
        x = torch.sigmoid(self.qlayer(x))
        '''
        print(x.shape)
        x = torch.relu(self.clayer_1(x))
        x = torch.sigmoid(self.clayer_2(x))
        '''
        return x
 
# Instantiate the model
model = HybridModel(n_qubits=2, n_layers=2)
 
# Dummy dataset: XOR logic gate
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0,1,1,0], dtype=torch.float32)
 
# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# Training loop
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")