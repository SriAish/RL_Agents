#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import count

import torch
import torch.optim as optim
from torch.distributions import normal


class GaussianAgent():
    def __init__(self, policy, mi, ma):
        self.policy = policy
        self.gamma = 0.01
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.min = mi[0]
        self.max = ma[0]
        
        # storage
        self.log_probs = []
        self.rewards = []
                
    def policy_update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/returns.std()
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.log_probs[:]
        
    def select_action(self, state, learn=False):
        state = torch.from_numpy(state.reshape(1, 3)).float()
        mean, std = self.policy(state)
        n = normal.Normal(mean, std)
        action = n.sample()
        if learn:
            self.log_probs.append(n.log_prob(action))
        return action