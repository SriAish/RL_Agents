import torch
import torch.nn as nn

class PendulumPolicy(nn.Module):
    def __init__(self, input_size, output_size):
        super(PendulumPolicy, self).__init__()
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
        
        # linear function read out
        self.fc4 = nn.Linear(8, output_size)
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
        
        # linear function
        out = self.fc4(out)
        # non linearity
        out = self.sigmoid(out)
        
        return out

def learn_pendulum_policy(env, agent, episodes=1000):
#     rewards = []
    for i_episode in range(episodes):
        state = env.reset()
#         ep_reward = 0
        for t in range(1, 10000):  
            action = agent.select_action(state, learn=True)
            state, reward, done, _ = env.step(action)
            agent.rewards.append(reward)
#             ep_reward += reward
            if done:
                break

        agent.policy_update()
#         if (i_episode + 1) % 20 == 0:
#             print(i_episode + 1, ep_reward)
        
    return agent