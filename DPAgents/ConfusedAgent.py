import numpy as np

class ConfusedAgent():
    """Class of a general discrete agent"""

    def __init__(self, env):
        self.env = env
        self.policy = np.ones(len(env.actions))/len(env.actions)

    def update(self):
        pass

    def get_action(self, state):
        return np.random.choice(np.arange(len(self.policy)), p=self.policy)

