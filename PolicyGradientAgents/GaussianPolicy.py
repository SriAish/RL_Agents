import torch
import torch.nn as nn

class GaussianPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(GaussianPolicy, self).__init__()
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
        self.mean = nn.Linear(8, output_size)
        # critic's layer
        self.std = nn.Linear(8, output_size)
        
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
        mean = self.mean(out)
        
        # linear function
        std = self.std(out)
        
        return mean, std
    