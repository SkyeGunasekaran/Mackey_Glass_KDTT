import numpy as np
import torch
import math
from jitcdde import jitcdde, y, t, jitcdde_lyap
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import RNN, MackeyGlass, test_model, LSTM

#alpha = 0.1510968296615217

mg = MackeyGlass(tau=17,
                 constant_past=0.9,
                 nmg = 10,
                 beta = 0.2,
                 gamma = 0.1,
                 dt=1.0,
                 splits=(1000., 250.),
                 start_offset=0.,
                 seed_id=0,)

train_set = Subset(mg, mg.ind_train)
test_set = Subset(mg, mg.ind_test)
train_set_new = []
test_set_new = []
#forcasting horizon of 1 is baseline MSE task
forcasting_horizon = 5

index = 0
for i in range(len(train_set)-forcasting_horizon):
    #sample = train_set[i][0]
    target = train_set[i+forcasting_horizon][1]
    train_set_new.append((torch.tensor([[i]], dtype=float), target))
    index+=1
for i in range(len(test_set)-forcasting_horizon):
    #sample = test_set[i][0]
    target = test_set[i+forcasting_horizon][1]
    test_set_new.append((torch.tensor([[i]], dtype=float), target))
    index+=1
# Hyperparameters
input_size = 1  # Depends on your dataset
hidden_size = 128  # Can be adjusted
output_size = 1  # Depends on your dataset
num_layers = 1  # Can be adjusted

# Instantiate the model
student_baseline = RNN(input_size, hidden_size, output_size)
#Baslineline Student Model
# Loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(student_baseline.parameters(), lr=0.001)
train_loader = DataLoader(train_set_new, batch_size=1, shuffle=False) # Adjust batch_size as needed
test_loader = DataLoader(test_set_new, batch_size=len(test_set_new), shuffle=False)  # Adjust batch_size as needed

# Number of epochs
num_epochs = 25  # Can be adjusted
for epoch in range(num_epochs):
    student_baseline.train()
    for inputs, targets in train_loader:
        inputs = inputs.float()
        targets = targets.float()
        outputs, _ = student_baseline(inputs)
        #print(torch.subtract(outputs, targets))
        #print(outputs, targets)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #test_model(student_baseline, test_loader, criterion=criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
student_baseline.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        inputs = inputs.float()  # Convert inputs to torch.float32
        targets = targets.float()  # Convert targets to torch.float32 if necessary
        outputs, _ = student_baseline(inputs)
        loss = torch.sqrt(criterion(outputs, targets))
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader):.4f}')
# Convert tensors to numpy arrays
targets_np = targets.cpu().detach().numpy()  # Use .cpu() if tensors are on GPU
outputs_np = outputs.cpu().detach().numpy()
# Create a time axis, assuming sequential time points
time_axis = np.arange(len(targets_np))
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_axis, targets_np, label='Targets', color='blue')
plt.plot(time_axis, outputs_np, label='Outputs', color='orange', linestyle='--')
plt.title('Targets vs Outputs')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
