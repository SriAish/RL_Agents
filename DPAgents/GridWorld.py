import numpy as np

class GridWorld():
    """Discrete GridWorld environment representation"""
    def __init__(self):
        """Initialize the environment"""
        self.world = np.zeros((8,8))

        # Rewards
        self.rewards = {'invalid_move': -2, 'terminated': 20, 'valid_move': -1, 'done': 0}

        # Allowed actions
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # Number of Actions
        self.nA = len(self.actions)

        # Number of states
        self.nS = 64

        # States
        self.states = [(i, j) for i in range(8) for j in range(8)]

        # Initially not terminated
        self.is_terminated = False

        # Ending point
        self.end = [(4, 0), (7, 7), (0, 6)]

        self.blocks = []

        # Marking 8 bricks in the gridworld
        self.blocks = [(0, 5), (1, 5), (1, 2), (4, 2), (4, 3), (5, 3), (6, 3), (7, 3)]
        for i in self.blocks:
            self.world[i] = 1

        # next state, reward, done for each action in each state
        self.P = []

        for i in range(self.nS):
            self.P.append([])
            for j in range(self.nA):
                self.P[i].append([])
                self.P[i][j] = []
                self.curr_pos = np.unravel_index(i, (8, 8))
                temp = list(self.move(j-1))
                temp.append(0.1)
                self.P[i][j].append(temp)
                self.curr_pos = np.unravel_index(i, (8, 8))
                temp = list(self.move(j))
                temp.append(0.8)
                self.P[i][j].append(temp)
                self.curr_pos = np.unravel_index(i, (8, 8))
                temp = list(self.move((j+1)%(self.nA - 1)))
                temp.append(0.1)
                self.P[i][j].append(temp)

        # Starting point
        self.curr_pos = (0, 0)


    def move(self, action):
        new_pos = tuple(np.array(self.curr_pos) + np.array(self.actions[action]))

        # If invalid move (i.e. going out of a grid or hittng a brick), position is unchanged and reward = -1
        if not self.valid_pos(new_pos):
            return (int)(self.states.index(self.curr_pos)), self.rewards['invalid_move'], self.is_terminated

        # If a valid move current poition changed to the new position.
        self.curr_pos = new_pos

        # If terminated reward of 10
        if self.check_terminated():
            return (int)(self.states.index(self.curr_pos)), self.rewards['terminated'], self.is_terminated

        return (int)(self.states.index(self.curr_pos)), self.rewards['valid_move'], self.is_terminated

    def step(self, action):
        prob = [0.1, 0.8, 0.1]
        action = (action + np.random.choice(len(prob), p=prob) - 1)%(self.nA - 1)
        return self.move(action)

    def reset(self):
        self.curr_pos = (0, 0)
        self.is_terminated = False
        return 0

    def valid_pos(self, pos):
        if pos[0] < 0 or pos[0] > 7:
            return False
        if pos[1] < 0 or pos[1] > 7:
            return False
        if pos in self.blocks:
            return False

        return True

    def check_terminated(self):
        if self.curr_pos in self.end:
            self.is_terminated = True
            return True
        return False
