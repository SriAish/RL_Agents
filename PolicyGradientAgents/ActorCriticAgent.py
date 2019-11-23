#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticAgent():
    def __init__(self, policy):
        self.policy = policy
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        
        # storage
        self.saved_actions = []
        self.rewards = []
                
    def policy_update(self):
        R = 0
        policy_loss = []
        value_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = returns - returns.mean()
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()

            # actor loss 
            policy_loss.append(-log_prob * advantage)

            # critic loss
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        
    def select_action(self, state, learn=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        
        if learn:
            self.saved_actions.append((m.log_prob(action), state_value))
        
        return action.item()