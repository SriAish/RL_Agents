#!/usr/bin/env python
import random
import time
from collections import defaultdict
from itertools import product
import numpy as np
import dill
import os
from env import TicTacToeEnv, agent_by_mark, next_mark, after_action_state

class MCOPA():
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    def __init__(self, mark, epsilon=0.1):
        self.mark = mark
        self.epsilon = epsilon

    def update(self, state, G):
        MCOPA.returns_sum[state] += G
        MCOPA.returns_count[state] += 1
        MCOPA.V[state] = MCOPA.returns_sum[state]/MCOPA.returns_count[state]

    def act(self, state, actions):
        e = random.random()
        if e < self.epsilon:
            return random.choice(actions)

        values = []
        for action in actions:
            next_state = after_action_state(state, action)
            next_val = 0

            if next_state in MCOPA.V:
                next_val = MCOPA.V[next_state]

            values.append(next_val)

        # select most right action for 'O' or 'X'
        if self.mark == 'O':
            ind = np.argmax(values)
        else:
            ind = np.argmin(values)

        action = actions[ind]
        return action

    def num_states(self):
        return len(MCOPA.V)

    def save(self):
        """save class as self.name.txt"""
        with open('onPolicyMC.dat','wb') as fh:
            dill.dump(self.__class__.__dict__, fh)

    def load(self):
        """try load self.name.txt"""
        if os.path.exists('onPolicyMC.dat'):
            with open('onPolicyMC.dat','rb') as fh:
                temp = dill.load(fh)

            MCOPA.V = temp['V']
            MCOPA.returns_sum = temp['returns_sum']
            MCOPA.returns_count = temp['returns_count']
            
    def reset(self):
        MCOPA.V = defaultdict(float)
        MCOPA.returns_sum = defaultdict(float)
        MCOPA.returns_count = defaultdict(float)
            
def learn_on_policy(episodes, epsilon=0.1, discount_factor=0.9):
    env = TicTacToeEnv()
    agents = [MCOPA('O', epsilon),
              MCOPA('X', epsilon)]

    start_mark = 'O'
    env.set_start_mark(start_mark)
    for i in range(episodes):
        episode = i + 1
        state = env.reset()
        _, mark = state
        steps = []
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            actions = env.available_actions()
            action = agent.act(state, actions)
            next_state, reward, done, _ = env.step(action)
            steps.append((state, reward))
            if done:
                break
            _, mark = state = next_state
            
        steps.reverse()
        G = 0
        # As in one episode of tic tac toe there will only be unique states we don't need to check for them
        for step in steps:
            _, mark = step[0]
            G = step[1] + discount_factor*G
            agents[0].update(step[0], G)
            
        # rotate start
        start_mark = next_mark(start_mark)
