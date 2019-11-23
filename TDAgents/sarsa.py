import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import dill
import os

from collections import defaultdict

class SARSA():
    def __init__(self, action_space, state_space, discount_factor=1.0, epsilon=0.1):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.action_space = action_space
        self.Q = dict([(x, [1, 1, 1, 1]) for x in range(state_space)])
        
    def policy(self, state):
        A = np.ones(self.action_space, dtype=float) * self.epsilon / self.action_space
        best_action = np.argmax(self.Q[state])
        A[best_action] += (1.0 - self.epsilon)
        return A
    
    def act(self, state):
        action_probs = self.policy(state)
        return np.random.choice(np.arange(len(action_probs)), p=action_probs)
    
    def get_value(self, state, action):
        return self.Q[state][action]
    
    def update(self, state, action, update):
        self.Q[state][action] += update
            
    def save(self, file):
        """save class as self.name.txt"""
        with open(file,'wb') as fh:
            dill.dump(self.__dict__, fh)

    def load(self, file):
        """try load self.name.txt"""
        if os.path.exists(file):
            with open(file,'rb') as fh:
                self.__dict__ = dill.load(fh)
            
    def reset(self):
        self.V = defaultdict(float)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        
def learn_sarsa(num_episodes, agent, env, discount_factor=0.99, alpha=0.05):
    for i in range(num_episodes):
        state = env.reset()
        action = agent.act(state)

        for t in range(2500):
            next_state, reward, done, info = env.step(action)
            next_action = agent.act(next_state)
            
            q_state = agent.get_value(state, action)
            if done:
                G = alpha * (reward - q_state)
            else:
                G = alpha * (reward + discount_factor * agent.get_value(next_state, next_action) - q_state)

            agent.update(state, action, G)

            state = next_state
            action = next_action

            if done:
                break
    return agent

def expected_value(q_sa):
    A = np.ones(4, dtype=float) * 0.1 / 4
    best_action = np.argmax(q_sa)
    A[best_action] += (1.0 - 0.1)
    return np.sum(A*q_sa)

def learn_esarsa(num_episodes, agent, env, discount_factor=0.99, alpha=0.05):
    for i in range(num_episodes):
        state = env.reset()
        action = agent.act(state)

        for t in range(2500):
            next_state, reward, done, info = env.step(action)
            next_action = agent.act(next_state)
            
            q_state = agent.get_value(state, action)
            if done:
                G = alpha * (reward - q_state)
            else:
                G = alpha * (reward + discount_factor * expected_value(agent.Q[next_state]) - q_state)

            agent.update(state, action, G)

            state = next_state
            action = next_action

            if done:
                break
    return agent