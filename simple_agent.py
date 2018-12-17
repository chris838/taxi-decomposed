import numpy as np
from collections import defaultdict
import random, sys, math, gym, pdb

class SimpleAgent:
    """
    Simple, flat Q-learning agent with no prior awareness of problem hierarchy
    """

    def __init__(self, states_shape=(500,), nA=6):

        self.nA = nA
        self.Q = np.zeros( states_shape + (nA,) )

        # Learning rate / step size
        self.alpha = 0.1
        self.alpha_decay = 0.99999
        self.alpha_min = 0.01

        # Discount
        self.gamma = 1
        self.gamma_decay = 1
        self.gamma_min = 1

        # Exploration
        self.epsilon = 0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0

        print("alpha: {0}, alpha_decay: {1}, alpha_min: {2}".format(
            self.alpha, self.alpha_decay, self.alpha_min))
        print("gamma: {0}, gamma_decay: {1}, gamma_min: {2}".format(
            self.gamma, self.gamma_decay, self.gamma_min))
        print("epsilon: {0}, epsilon_decay: {1}, epsilon_min: {2}".format(
            self.epsilon, self.epsilon_decay, self.epsilon_min))

    def select_action(self, state):

        # Epsilon-greedy action selection
        Q = self.Q
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.nA)
        else:
            return np.argmax(Q[state])

    def step(self, state, action, reward, next_state, done):

        # Q-learning
        Q = self.Q  # alias
        target = reward + self.gamma * Q[next_state][np.argmax(Q[next_state])]
        Q[state][action] += self.alpha * (target - Q[state][action])

        # Decay exploration, step size and discount over time
        self.alpha = min(1, max(self.alpha_min, self.alpha * self.alpha_decay))
        self.epsilon = min(
            1, max(self.epsilon_min, self.epsilon * self.epsilon_decay))
        self.gamma = min(1, max(self.gamma_min, self.gamma * self.gamma_decay))
