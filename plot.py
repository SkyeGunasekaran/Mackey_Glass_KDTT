import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step for regression
        return out
from utils import MackeyGlass, Subset
mg = MackeyGlass(tau=17,
                 constant_past=0.9,
                 nmg = 10,
                 beta = 0.2,
                 gamma = 0.1,
                 dt=1.0,
                 splits=(5000., 250.),
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

# Assuming your data is a list of tuples (x, y), where x and y are tensors
# Convert your data to PyTorch tensors
x_data = [sample[0] for sample in train_set_new]
y_data = [sample[1] for sample in train_set_new]
x_data = torch.stack(x_data).to(torch.float32)
y_data = torch.stack(y_data).to(torch.float32)

# Instantiate the model
input_size = x_data.size(-1)  # Assuming the size of the last dimension of x is the input size
output_size = y_data.size(-1)  # Assuming the size of the last dimension of y is the output size
hidden_size = 64  # Adjust this based on the complexity of your problem

model = SimpleRNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000  # Adjust as needed

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_data)

    # Compute loss
    loss = criterion(outputs, y_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the model to make predictions
with torch.no_grad():
    test_input = torch.tensor(...)  # Provide your test input data
    predicted_output = model(test_input.unsqueeze(0))  # Add batch dimension
    print(predicted_output)
