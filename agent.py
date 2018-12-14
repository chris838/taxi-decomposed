import numpy as np
from collections import deque, defaultdict
import random, sys, math, gym

class Agent:

    def __init__(self, nA=6):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.8
        self.episode = 1

    def select_action(self, state, i_episode, num_episodes):
        if i_episode != self.episode and i_episode % (num_episodes * 0.0005) == 0:
            self.epsilon -= 50/num_episodes

        if i_episode != self.episode and i_episode % (num_episodes * 0.005) == 0:
            self.alpha -= 20/num_episodes
            self.epsilon = 0.8

        if i_episode != self.episode and i_episode % (num_episodes * 0.05) == 0:
            self.gamma -= 20/num_episodes
            self.epsilon = 0.8
            self.alpha = 0.8

        return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):
        next_action = np.argmax(self.Q[state])
        self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + \
            (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]))
