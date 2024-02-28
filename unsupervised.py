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
from utils import RNN, MackeyGlass, test_model
import torch.nn.functional as F
torch.cuda.empty_cache()
# Hyperparameters
input_size = 1  # Depends on your dataset
hidden_size = 128  # Can be adjusted
output_size = 1  # Depends on your dataset
num_layers = 1  # Can be adjusted
epochs = 10
alpha = 0.5#0.1510968296615217
temperature = 4
torch.cuda.empty_cache()

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
train_set = list(train_set)
test_set = list(test_set)
train_set_new = []
test_set_new = []
#forcasting horizon of 1 is baseline MSE task

forcasting_horizon = 5
student_helper = train_set[forcasting_horizon:]
for i in range(len(train_set)-forcasting_horizon):
    sample, target = train_set[i]
    target = train_set[i+forcasting_horizon][1]
    train_set_new.append((sample, target))
for i in range(len(test_set)-forcasting_horizon):
    sample, target = test_set[i]
    target = test_set[i+forcasting_horizon][1]
    #print(sample, target)
    test_set_new.append((sample, target))
# Instantiate the model
teacher = RNN(input_size, hidden_size, output_size, num_layers)
student = RNN(input_size, hidden_size, output_size, num_layers)

window_difference = 0
diff_train = len(train_set) - len(train_set_new)
for i in range(diff_train+window_difference):
    train_set_new.insert(0, (-1, -1))
for i in range(window_difference):
    train_set.append((-1, -1))
for i in range(forcasting_horizon):
    student_helper.insert(0, (-1, -1))

# Loss and optimizer
criterion = nn.MSELoss()
optimizer_teacher = optim.Adam(teacher.parameters(), lr=0.001)
optimizer_student = optim.Adam(student.parameters(), lr=0.001)
student_helper = DataLoader(student_helper, batch_size=1, shuffle=False)
train_loader_teacher = DataLoader(train_set, batch_size=1, shuffle=False) # Adjust batch_size as needed
test_loader_teacher = DataLoader(test_set, batch_size=len(test_set), shuffle=False)  # Adjust batch_size as needed
train_loader_student = DataLoader(train_set_new, batch_size=1, shuffle=False) # Adjust batch_size as needed
test_loader_student = DataLoader(test_set_new, batch_size=len(test_set_new), shuffle=False)  # Adjust batch_size as needed
# Number of epochs
kl_div = nn.KLDivLoss
stopping_epoch = 1

for epoch in range(epochs):
    teacher.train()
    student.train()
    loss_1_list = []
    loss_2_list = []
    for i, ((inputs_teacher, targets_teacher), (inputs_student, targets_student), (inputs_helper, targets_helper)) in enumerate(zip(train_loader_teacher, train_loader_student, student_helper)): 
        #if torch.equal(targets_teacher,targets_student):
        #    print(i)
        inputs_helper = inputs_helper.float()
        targets_helper = targets_helper.float()
        inputs_teacher = inputs_teacher.float()
        inputs_student = inputs_student.float()
        targets_teacher = targets_teacher.float()
        targets_student = targets_student.float()
        #print(targets_helper, targets_teacher)
        if epoch < stopping_epoch:
            if not torch.any(inputs_teacher == -1.0):
                outputs_teacher, _ = teacher(inputs_teacher)
                loss_teacher = criterion(outputs_teacher, targets_teacher)
                optimizer_teacher.zero_grad()
                loss_teacher.backward()
                optimizer_teacher.step()

        if not torch.any(inputs_student == -1.0):
            outputs_student, student_logits = student(inputs_student)
            teacher.eval()
            with torch.no_grad():
                _, teacher_logits = teacher(inputs_helper)
                assert(torch.equal(targets_helper, targets_student))
            loss_1 = alpha*criterion(student_logits, teacher_logits)
            loss_2 = (1-alpha)*criterion(outputs_student, targets_student)
            loss_1_list.append(loss_1)
            loss_2_list.append(loss_2)
            #print(loss_1, loss_2)
            loss_student = loss_1+loss_2
            #loss_student = alpha*F.kl_div(F.log_softmax(outputs_student/temperature, dim=1), F.softmax(teacher_logits/temperature, dim=1), reduction='batchmean') + (1-alpha)*criterion(outputs_student, targets_student)
            optimizer_student.zero_grad()
            loss_student.backward()
            optimizer_student.step()
            teacher.train()
    print(f"Average loss of KD for epoch {epoch}: {sum(loss_1_list)/len(loss_1_list):.4f}")
    print(f"Average loss of MSE for epoch {epoch}: {sum(loss_2_list)/len(loss_2_list):.4f}")
        
    #test_model(student, test_loader_student, criterion=criterion)
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


teacher.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader_teacher:
        inputs = inputs.float()  # Convert inputs to torch.float32
        targets = targets.float()  # Convert targets to torch.float32 if necessary
        outputs, _ = teacher(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader_student):.4f}')

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

student.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader_student:
        inputs = inputs.float()  # Convert inputs to torch.float32
        targets = targets.float()  # Convert targets to torch.float32 if necessary
        outputs, _ = student(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    end_loss = total_loss/len(test_loader_student)
    print(f'Test Loss: {end_loss:.4f}')

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
    
'''
# Extract parameters from the models
params1 = list(student.parameters())
params2 = list(teacher.parameters())

# Flatten and concatenate parameters for comparison
params_flat1 = torch.cat([p.view(-1) for p in params1])
params_flat2 = torch.cat([p.view(-1) for p in params2])

# Calculate similarity metric (cosine similarity or Frobenius norm)
cosine_similarity = torch.nn.functional.cosine_similarity(params_flat1, params_flat2, dim=0)
frobenius_norm = torch.norm(params_flat1 - params_flat2)

print("Cosine Similarity:", cosine_similarity.item())
print("Frobenius Norm:", frobenius_norm.item())
'''