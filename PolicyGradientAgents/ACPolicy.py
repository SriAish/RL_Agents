import torch
import torch.nn as nn

class ACPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ACPolicy, self).__init__()
        # linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # dropout
        self.drop = nn.Dropout(p=0.6)
        # non linearity
        self.relu = nn.ReLU()
        # actor's layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # critic's layer
        self.value_head = nn.Linear(hidden_dim, 1)
        # non linearity
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # linear function
        out = self.fc1(x)
        # dropout
        out = self.drop(out)
        # non linearity
        out = self.relu(out)

        # state value
        state_values = self.value_head(out)

        # linear function
        out = self.fc2(out)
        # non linearity
        out = self.sigmoid(out)
        
        return out, state_values