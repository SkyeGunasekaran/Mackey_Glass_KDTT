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
import optuna
class MackeyGlass(Dataset):
    """ Dataset for the Mackey-Glass task.
    """
    def __init__(self,
                 tau,
                 constant_past,
                 nmg = 10,
                 beta = 0.2,
                 gamma = 0.1,
                 dt=1.0,
                 splits=(8000., 2000.),
                 start_offset=0.,
                 seed_id=0,
    ):
        """
        Initializes the Mackey-Glass dataset.

        Args:
            tau (float): parameter of the Mackey-Glass equation
            constant_past (float): initial condition for the solver
            nmg (float): parameter of the Mackey-Glass equation
            beta (float): parameter of the Mackey-Glass equation
            gamma (float): parameter of the Mackey-Glass equation
            dt (float): time step length for sampling data
            splits (tuple): data split in time units for training and testing data, respectively
            start_offset (float): added offset of the starting point of the time-series, in case of repeating using same function values
            seed_id (int): seed for generating function solution
        """

        super().__init__()

        # Parameters
        self.tau = tau
        self.constant_past = constant_past
        self.nmg = nmg
        self.beta = beta
        self.gamma = gamma
        self.dt = dt

        # Time units for train (user should split out the warmup or validation)
        self.traintime = splits[0]
        # Time units to forecast
        self.testtime = splits[1]

        self.start_offset = start_offset
        self.seed_id = seed_id

        # Total time to simulate the system
        self.maxtime = self.traintime + self.testtime + self.dt

        # Discrete-time versions of the continuous times specified above
        self.traintime_pts = round(self.traintime/self.dt)
        self.testtime_pts = round(self.testtime/self.dt)
        self.maxtime_pts = self.traintime_pts + self.testtime_pts + 1 # eval one past the end

        # Specify the system using the provided parameters
        self.mackeyglass_specification = [ self.beta * y(0,t-self.tau) / (1 + y(0,t-self.tau)**self.nmg) - self.gamma*y(0) ]

        # Generate time-series
        self.generate_data()

        # Generate train/test indices
        self.split_data()


    def generate_data(self):
        """ Generate time-series using the provided parameters of the equation.
        """
        np.random.seed(self.seed_id)

        # Create the equation object based on the settings
        self.DDE = jitcdde_lyap(self.mackeyglass_specification)
        self.DDE.constant_past([self.constant_past])
        self.DDE.step_on_discontinuities()

        ##
        ## Generate data from the Mackey-Glass system
        ##
        self.mackeyglass_soln = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        lyaps_weights = torch.zeros((self.maxtime_pts,1),dtype=torch.float64)
        count = 0
        for time in torch.arange(self.DDE.t+self.start_offset, self.DDE.t+self.start_offset+self.maxtime, self.dt,dtype=torch.float64):
            value, lyap, weight = self.DDE.integrate(time.item())
            self.mackeyglass_soln[count,0] = value[0]
            lyaps[count,0] = lyap[0]
            lyaps_weights[count,0] = weight
            count += 1

        # Total variance of the generated Mackey-Glass time-series
        self.total_var=torch.var(self.mackeyglass_soln[:,0], True)

        # Estimate Lyapunov exponent
        self.lyap_exp = ((lyaps.T@lyaps_weights)/lyaps_weights.sum()).item()


    def split_data(self):
        """ Generate training and testing indices.
        """
        self.ind_train = torch.arange(0, self.traintime_pts)
        self.ind_test = torch.arange(self.traintime_pts, self.maxtime_pts-1)

    def __len__(self):
        """ Returns number of samples in dataset.

        Returns:
            int: number of samples in dataset
        """
        return len(self.mackeyglass_soln)-1

    def __getitem__(self, idx):
        """ Getter method for dataset.

        Args:
            idx (int): index of sample to return

        Returns:
            sample (tensor): individual data sample, shape=(timestamps, features)=(1,1)
            target (tensor): corresponding next state of the system, shape=(label,)=(1,)
        """
        sample = torch.unsqueeze(self.mackeyglass_soln[idx, :], dim=0)
        target = self.mackeyglass_soln[idx+1, :]

        return sample, target
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)

        # Pass through the linear layer
        out = self.fc(out[:, -1, :])
        return out

def objective(trial):
    #alpha = 0.1510968296615217
    torch.cuda.empty_cache()
    alpha = trial.suggest_float('alpha', 0.1, 1)
    mg = MackeyGlass(tau=17,
                     constant_past=0.9,
                     nmg = 10,
                     beta = 0.2,
                     gamma = 0.1,
                     dt=1.0,
                     splits=(8000., 2000.),
                     start_offset=0.,
                     seed_id=0,)

    train_set = Subset(mg, mg.ind_train)
    test_set = Subset(mg, mg.ind_test)
    train_set_new = []
    test_set_new = []

    #forcasting horizon of 1 is baseline MSE task
    forcasting_horizon = 20
    for i in range(len(train_set)-forcasting_horizon):
        sample, target = train_set[i]
        target = train_set[i+forcasting_horizon][1]
        train_set_new.append((sample, target))

    for i in range(len(test_set)-forcasting_horizon):
        sample, target = test_set[i]
        target = test_set[i+forcasting_horizon][1]
        #print(sample, target)
        test_set_new.append((sample, target))

    # Hyperparameters
    input_size = 1  # Depends on your dataset
    hidden_size = 128  # Can be adjusted
    output_size = 1  # Depends on your dataset
    num_layers = 1  # Can be adjusted

    '''
    # Instantiate the model
    student_baseline = RNN(input_size, hidden_size, output_size, num_layers)

    #Baslineline Student Model

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_baseline.parameters(), lr=0.001)

    train_loader = DataLoader(train_set_new, batch_size=10, shuffle=False) # Adjust batch_size as needed
    test_loader = DataLoader(test_set_new, batch_size=len(train_set_new), shuffle=False)  # Adjust batch_size as needed
    print(len(train_loader))

    # Number of epochs
    num_epochs = 10  # Can be adjusted

    for epoch in range(num_epochs):
        student_baseline.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float()
            targets = targets.float()
            outputs = student_baseline(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    student_baseline.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.float()  # Convert inputs to torch.float32
            targets = targets.float()  # Convert targets to torch.float32 if necessary

            outputs = student_baseline(inputs)
            loss = criterion(outputs, targets)
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
    '''
    #Training the teacher model

    # Instantiate the model
    teacher = RNN(input_size, hidden_size, output_size, num_layers)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)

    train_loader = DataLoader(train_set, batch_size=10, shuffle=False) # Adjust batch_size as needed
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)  # Adjust batch_size as needed

    # Number of epochs
    num_epochs = 10  # Can be adjusted

    for epoch in range(num_epochs):
        teacher.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float()
            targets = targets.float()
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    teacher.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.float()  # Convert inputs to torch.float32
            targets = targets.float()  # Convert targets to torch.float32 if necessary
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

        #print(f'Test Loss: {total_loss / len(test_loader):.4f}')

    '''
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


    #Training the student with distillation
    # Instantiate the model
    student = RNN(input_size, hidden_size, output_size, num_layers)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    train_loader = DataLoader(train_set_new, batch_size=10, shuffle=False) # Adjust batch_size as needed
    test_loader = DataLoader(test_set_new, batch_size=len(test_set_new), shuffle=False)  # Adjust batch_size as needed

    # Number of epochs
    num_epochs = 10  # Can be adjusted

    for epoch in range(num_epochs):
        student.train()
        teacher.eval()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float()
            targets = targets.float()
            outputs = student(inputs)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            loss = alpha*criterion(outputs, teacher_logits) + (1-alpha)*criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    student.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            inputs = inputs.float()  # Convert inputs to torch.float32
            targets = targets.float()  # Convert targets to torch.float32 if necessary
            outputs = student(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        end_loss = total_loss/len(test_loader)
        #print(f'Test Loss: {end_loss:.4f}')
    return end_loss
    '''
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
def main():
    # Load existing study or create a new one
    study_name = "Mackey_Glass_Test"
    storage_name = f"sqlite:///optuna_{study_name}.db"

    try:
        study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name)
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(study_name, storage=storage_name)

    study.optimize(objective, n_trials=1000)  # Optimize one trial at a time

    # Save the study history to a file after each trial
    df = study.trials_dataframe()
    df.to_csv(f"optuna_{study_name}_history.csv", index=False)
        
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == '__main__':
    main()