import numpy as np

class Cliff:

    def __init__(self, start, goal, rows, cols, obstacles):
        self.start = start
        self.goal  = goal
        self.rows  = rows
        self.cols  = cols

        # cliffs normal space are marked sa 0
        self.board = np.zeros([rows, cols])
        self.state = start
        self.actions = ["up", "left", "right", "down"]

        # mud places are marked as -1
        for i, j in obstacles:
            self.board[i, j] = -1

        # goal are marked as 1
        self.board[self.goal[0], self.goal[1]] = 1

        # start is marked as 2
        self.board[self.start[0], self.start[1]] = 2

    def setState(self, state):
        self.state = state

    def transition(self, action):

        self.state = self.getNextState(action)
        reward = self.getReward(self.state)

        return self.state, reward

    def getNextState(self, action):
        # Calculate next position
        if action == "up":
            nxtPos = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtPos = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtPos = (self.state[0], self.state[1] - 1)
        elif action == "right":
            nxtPos = (self.state[0], self.state[1] + 1)

        # check space limits
        if nxtPos[0] >= 0 and nxtPos[0] <= self.rows-1 and\
            nxtPos[1] >= 0 and nxtPos[1] <= self.cols-1:
            state = nxtPos
            return state

        if self.board[nxtPos] == -2:
            return self.start

        return state

    def getReward(self, state):
        # check space limits
        if state[0] >= 0 and state[0] <= self.rows-1 and\
           state[1] >= 0 and state[1] <= self.cols-1:
            reward = 0

        # get reward according to state
        if self.board[state] == 0:
            reward = -2
        elif state == self.goal:
            reward = 1000
        elif state == self.start:
            reward = 0
        else:
            reward = -1000

        return reward

    def getState(self, state):
        return self.state

    def reset(self):
        self.state = self.start

    def show(self):
        for i in range(0, self.rows):
            print('-------------------------------------------------')
            out = '| '
            for j in range(0, self.cols):
                if self.board[i, j] == -1:
                    token = '*'
                if self.board[i, j] == 0:
                    token = '0'
                if (i, j) == self.start:
                    token = 'S'
                if (i, j) == self.goal:
                    token = 'G'
                if (i, j) == self.state:
                    token = 'P'
                out += token + ' | '
            print(out)
        print('-------------------------------------------------')
