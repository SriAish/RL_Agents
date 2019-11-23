import numpy as np
from GridWorld import GridWorld

class PolicyIteration():
    """Class of a general discrete agent"""

    def __init__(self, env, discount_factor=0.99, theta=0.000001):
        self.env = env
        self.valueFunction = np.zeros(self.env.nS)
        self.discount_factor = discount_factor
        self.theta = theta
        self.policy = np.ones([self.env.nS, self.env.nA])/self.env.nA
        self.sweeps = 0

    def evaluate_policy(self, policy):
        # Start with a random (all 0) value function
        V = self.valueFunction
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(self.env.nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]):
                    # For each action, look at the possible next state.
                    for next_state, reward, done, prob in self.env.P[s][a]:
                        v += action_prob * prob * (reward + self.discount_factor * V[(int)(next_state)])

                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v

            # Stop evaluating once our value function change is below a threshold
            if delta < self.theta:
                break
        return np.array(V)

    def next_step_rewards(self, state, V):
        A = np.zeros(self.env.nA)
        for i in range(self.env.nA):
            for next_state, reward, done, prob in self.env.P[state][i]:
                A[i] += prob * (reward + self.discount_factor*V[(int)(next_state)])
        return A

    def update_policy(self):

        is_stable = True

        for i in range(self.env.nS):
            curr_best = np.argmax(self.policy[i])

            rewards = self.next_step_rewards(i, self.valueFunction)

            actual_best = np.argmax(rewards)

            if curr_best != actual_best:
                is_stable = False

            self.policy[i] =  np.eye(self.env.nA)[actual_best]

        return is_stable

    def get_policy(self):
        return np.argmax(self.policy, axis = 1)
    
    def update(self):
        self.sweeps += 1
        self.valueFunction = self.evaluate_policy(self.policy)
        return self.sweeps, self.update_policy()

    def get_action(self, state):
        return np.random.choice(np.arange(len(self.policy[state])), p=self.policy[state])

if __name__ == "__main__":
    PIA = PolicyIteration(GridWorld())
    x = False
    while not x:
        y, x = PIA.update()

    print(PIA.valueFunction.reshape(8,8))
