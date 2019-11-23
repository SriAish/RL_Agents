import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Policy, self).__init__()
        # linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # dropout
        self.drop = nn.Dropout(p=0.6)
        # non linearity
        self.relu = nn.ReLU()
        # linear function read out
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # non linearity
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # linear function
        out = self.fc1(x)
        # dropout
        out = self.drop(out)
        # non linearity
        out = self.relu(out)
        # linear function
        out = self.fc2(out)
        # non linearity
        out = self.sigmoid(out)
        
        return out

def learn_policy(env, agent, episodes=1000):
    for i_episode in range(episodes):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  
            action = agent.select_action(state, learn=True)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
            if done:
                break

        agent.policy_update()
        
    return agent