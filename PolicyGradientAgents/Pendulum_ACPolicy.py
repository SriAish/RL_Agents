import torch
import torch.nn as nn

class PendulumACPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(PendulumACPolicy, self).__init__()
        # linear function
        self.fc1 = nn.Linear(input_size, 8)
        # non linearity
        self.relu1 = nn.ReLU()
        
        # linear function
        self.fc2 = nn.Linear(8, 8)
        # non linearity
        self.relu2 = nn.ReLU()
        
        # linear function
        self.fc3 = nn.Linear(8, 8)
        # non linearity
        self.relu3 = nn.ReLU()
        
        # actor's layer
        self.fc4 = nn.Linear(8, output_size)
        # critic's layer
        self.value_head = nn.Linear(8, 1)
        # non linearity
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # linear function
        out = self.fc1(x)
        # non linearity
        out = self.relu1(out)
        
        # linear function
        out = self.fc2(out)
        # non linearity
        out = self.relu2(out)
        
        # linear function
        out = self.fc3(out)
        # non linearity
        out = self.relu3(out)
        
        # state value
        state_values = self.value_head(out)
        
        # linear function
        out = self.fc4(out)
        # non linearity
        out = self.sigmoid(out)
        
        return out, state_values
    