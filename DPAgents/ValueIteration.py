import numpy as np
from GridWorld import GridWorld

class ValueIteration():
    """Class of a general discrete agent"""

    def __init__(self, env, discount_factor=0.99, theta=0.000001):
        self.env = env
        self.valueFunction = np.zeros(self.env.nS)
        self.discount_factor = discount_factor
        self.theta = theta
        self.policy = np.zeros([self.env.nS, self.env.nA]) + 1/self.env.nA
        self.sweeps = 0

    def update(self):
        self.sweeps += 1
        max_diff = 0
        for s in range(self.env.nS):
            max_v = -999999
            max_a = 0
            for a in range(self.env.nA):
                v = 0;
                for next_state, reward, done, prob in self.env.P[s][a]:
                    v += prob * (reward + self.discount_factor * self.valueFunction[(int)(next_state)])
                if v > max_v:
                    max_a = a
                    max_v = max(v, max_v)

            delta = max(max_diff, np.abs(self.valueFunction[s]-max_v))
            self.valueFunction[s] = max_v
            self.policy[s] = np.eye(self.env.nA)[max_a]

        if delta < self.theta:
            return self.sweeps, True

        return self.sweeps, False
    
    def get_policy(self):
        return np.argmax(self.policy, axis = 1)

    def get_action(self, state):
        return np.random.choice(np.arange(len(self.policy[state])), p=self.policy[state])

if __name__ == "__main__":
    env = GridWorld()
    via = ValueIteration(env)
    x = False
    while not x:
        s, x = via.update()
    print(via.valueFunction.reshape(8,8))
