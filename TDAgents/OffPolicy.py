#!/usr/bin/env python
import random
import time
from collections import defaultdict
from itertools import product
import numpy as np
import dill
import os
from env import TicTacToeEnv, agent_by_mark, next_mark, after_action_state

class MCOffPA():
    V = defaultdict(float)
    C = defaultdict(float)

    def __init__(self, mark):
        self.mark = mark
        
    def update(self, state, G, W):
        MCOffPA.C[state] = MCOffPA.C[state] + W
        MCOffPA.V[state] = MCOffPA.V[state] + (W/MCOffPA.C[state])*(G - MCOffPA.V[state])
            
    def act(self, state, actions):
        values = []
        for action in actions:
            next_state = after_action_state(state, action)
            next_val = 0
            if next_state in MCOffPA.V:
                next_val = MCOffPA.V[next_state]
            values.append(next_val)
            
        # select most right action for 'O' or 'X'
        if self.mark == 'O':
            ind = np.argmax(values)
        else:
            ind = np.argmin(values)

        action = actions[ind]

        return action
    
    def num_states(self):
        return len(MCOffPA.V)
        
    def save(self):
        """save class as self.name.txt"""
        with open('offPolicyMC.dat','wb') as fh:
            dill.dump(self.__class__.__dict__, fh)

    def load(self):
        """try load self.name.txt"""
        if os.path.exists('offPolicyMC.dat'):
            with open('offPolicyMC.dat','rb') as fh:
                temp = dill.load(fh)
                
        MCOffPA.V = temp['V']
        MCOffPA.C = temp['C']
        
    def reset(self):
        MCOffPA.V = defaultdict(float)
        MCOffPA.C = defaultdict(float)

def learn_off_policy(episodes, discount_factor=0.9):
    env = TicTacToeEnv()
    agents = [MCOffPA('O'),
              MCOffPA('X')]

    start_mark = 'O'
    env.set_start_mark(start_mark)
    for i in range(episodes):
        state = env.reset()
        _, mark = state
        steps = []
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            actions = env.available_actions()
            action = random.choice(actions)
            next_state, reward, done, _ = env.step(action)
            steps.append((state, reward, action, actions))
            if done:
                break
            _, mark = state = next_state
            
        steps.reverse()
        G = 0
        W = 1
        
        # As in one episode of tic tac toe there will only be unique states we don't need to check for them
        for step in steps:
            _, mark = step[0]
            agent = agent_by_mark(agents, mark)
            G = step[1] + discount_factor*G
            agent.update(step[0], G, W)
            if agent.act(step[0], step[3]) != step[2]:
                break
                
            # behaviour policy = 1/available_actions
            W = W*len(step[3])
            
        # rotate start
        start_mark = next_mark(start_mark)