import numpy as np
from collections import defaultdict
import random, sys, math, gym

class DropPickAgent:
    """
    Simple, flat Q-learning agent with hard-coded heuristics for dropping off and
    picking up the passanger. This entirely avoids the penalties for illegally
    perfoming those actions.
    """

    def __init__(self, nA=6):

        self.nA = nA
        self.Q = defaultdict(lambda: np.ones(self.nA, dtype=np.float64) * 0.0)

        # Learning rate / step size
        self.alpha = 0.1
        self.alpha_decay = 0.99999
        self.alpha_min = 0.01

        # Discount
        self.gamma = 1
        self.gamma_decay = 1
        self.gamma_min = 0

        # Exploration
        self.epsilon = 0
        self.epsilon_decay = 1
        self.epsilon_min = 0

        # Environment priors
        self.action_pickup = 4
        self.action_dropoff = 5
        self.locs = [(0,0), (0,4), (4,0), (4,3)]
        self.passenger_in_taxi_index = 4

        print("alpha: {0}, alpha_decay: {1}, alpha_min: {2}".format(
            self.alpha, self.alpha_decay, self.alpha_min))
        print("gamma: {0}, gamma_decay: {1}, gamma_min: {2}".format(
            self.gamma, self.gamma_decay, self.gamma_min))
        print("epsilon: {0}, epsilon_decay: {1}, epsilon_min: {2}".format(
            self.epsilon, self.epsilon_decay, self.epsilon_min))

    def select_action(self, state):

        # Override epsilon-greedy exploration for pickup/dropoff
        if self.can_pick_up(state):
            return self.action_pickup
        if self.can_drop_off(state):
            return self.action_dropoff

        # Otherwise, perform epsilon-greedy action selection, but preclude
        # the pickup and dropoff actions
        Q = self.Q
        if np.random.rand() < self.epsilon:
            # truncate action space to preclude pickup/dropoff
            return np.random.choice(self.nA - 2)
        else:
            # truncate action space to preclude pickup/dropoff
            return np.argmax(Q[state][:-2])


    def can_pick_up(self, state):
        taxi_loc, pass_idx, dest_idx = self.decode_state(state)

        # Can't pickup whilst in taxi
        if pass_idx == self.passenger_in_taxi_index:
            return False
        # Otherwise, taxi must be colocated with passenger
        return taxi_loc == self.locs[pass_idx]


    def can_drop_off(self, state):
        taxi_loc, pass_idx, dest_idx = self.decode_state(state)

        # Can't dropoff whilst not in taxi
        if pass_idx != self.passenger_in_taxi_index:
            return False
        # Otherwise, taxi must be colocated with destination
        return taxi_loc == self.locs[dest_idx]


    def step(self, state, action, reward, next_state, done):

        # Q-learning
        Q = self.Q  # alias
        target = reward + self.gamma * Q[next_state][np.argmax(Q[next_state])]
        Q[state][action] += self.alpha * (target + Q[state][action])

        # Decay exploration, step size and discount over time
        self.alpha = min(1, max(self.alpha_min, self.alpha * self.alpha_decay))
        self.epsilon = min(
            1, max(self.epsilon_min, self.epsilon * self.epsilon_decay))
        self.gamma = min(1, max(self.gamma_min, self.gamma * self.gamma_decay))


    def decode_state(self, i):
        # returns (taxi_row, taxi_column), passenger_index, destination_index
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        taxirow, taxicol, pass_idx, dest_idx = reversed(out)
        taxiloc = (taxirow, taxicol)
        return taxiloc, pass_idx, dest_idx
