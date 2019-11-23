import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import dill
import os

from collections import defaultdict

class QL():
    def __init__(self, action_space, state_space, discount_factor=1.0):
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.Q = dict([(x, [1, 1, 1, 1]) for x in range(state_space)])
        
    def act(self, state):
        return np.argmax(self.Q[state])
    
    def get_value(self, state, action):
        return self.Q[state][action]
    
    def update(self, state, action, update):
        self.Q[state][action] += update
            
    def save(self):
        """save class as self.name.txt"""
        with open('SARSA.dat','wb') as fh:
            dill.dump(self.__dict__, fh)

    def load(self):
        """try load self.name.txt"""
        if os.path.exists('SARSA.dat'):
            with open('SARSA.dat','rb') as fh:
                self.__dict__ = dill.load(fh)
            
    def reset(self):
        self.V = defaultdict(float)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        
def epsilon_greedy_policy(state, Q):
    A = np.ones(4, dtype=float) * 0.1 / 4
    best_action = np.argmax(Q[state])
    A[best_action] += (1.0 - 0.1)
    return np.random.choice(np.arange(len(A)), p=A)

def learn_ql(num_episodes, agent, env, discount_factor=0.99, alpha=0.05):
    for i in range(num_episodes):
        state = env.reset()
        
        for t in range(2500):
            action = epsilon_greedy_policy(state, agent.Q)
            next_state, reward, done, _ = env.step(action)
            best_action = agent.act(next_state)
            
            q_state = agent.get_value(state, action)
            if done:
                G = alpha * (reward - q_state)
            else:
                G = alpha * (reward + discount_factor * agent.get_value(next_state, best_action) - q_state)

            agent.update(state, action, G)

            state = next_state

            if done:
                break
    return agent