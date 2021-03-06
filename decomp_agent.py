import numpy as np
from collections import defaultdict
from simple_agent import SimpleAgent
import random, sys, math, gym, pdb


class DecompAgent:
    """
    This agent takes advantage of the problem sub-structure by decomposing the
    root problem into a navigation subproblem (which it solves using a simple
    Q-learning agent) and using hand-crafted heuristics for all other decisions.
    """

    def __init__(self):

        # Only four navigation actions for subproblem actions
        nA = 4
        # Only taxi row, column and destination index for subproblem states
        states_shape = (5, 5, 4)
        self.sub_agent = SimpleAgent(states_shape=states_shape, nA=nA)

        # Learning rate / step size
        self.sub_agent.alpha = 0.01
        self.sub_agent.alpha_decay = 1
        self.sub_agent.alpha_min = 0

        # Discount
        self.sub_agent.gamma = 1
        self.sub_agent.gamma_decay = 1
        self.sub_agent.gamma_min = 0

        # Exploration
        self.sub_agent.epsilon = 0.01
        self.sub_agent.epsilon_decay = 1
        self.sub_agent.epsilon_min = 0

        # For our params, just mimic the sub-agent's
        (self.alpha, self.epsilon, self.gamma) = \
            self.sub_agent.alpha, self.sub_agent.epsilon, self.sub_agent.gamma

        # Environment priors
        self.action_pickup = 4
        self.action_dropoff = 5
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.passenger_in_taxi_idx = 4

        print("alpha: {0}, alpha_decay: {1}, alpha_min: {2}".format(
            self.sub_agent.alpha, self.sub_agent.alpha_decay, self.sub_agent.alpha_min))
        print("gamma: {0}, gamma_decay: {1}, gamma_min: {2}".format(
            self.sub_agent.gamma, self.sub_agent.gamma_decay, self.sub_agent.gamma_min))
        print("epsilon: {0}, epsilon_decay: {1}, epsilon_min: {2}".format(
            self.sub_agent.epsilon, self.sub_agent.epsilon_decay, self.sub_agent.epsilon_min))

    def select_action(self, state):

        # Override epsilon-greedy exploration for pickup/dropoff
        if self.can_pick_up(state):
            return self.action_pickup
        if self.can_drop_off(state):
            return self.action_dropoff

        # Otherwise, defer to the sub-agent
        transformed_state = self.transform_state(state)
        return self.sub_agent.select_action(transformed_state)

    def step(self, state, action, reward, next_state, done):
        # Transform experience into the problem space of the sub-agent

        # If the selected action was pickup/dropoff, then experience is not
        # relevant to sub-problem
        if action == self.action_pickup or action == self.action_dropoff:
            return

        # If we can pickup/dropoff in the next state, then for the
        # sub-problem we consider next_state to be terminal and the episode
        # concluded
        if self.can_pick_up(next_state) or self.can_drop_off(next_state):
            state_t = self.transform_state(state)
            action_t = self.transform_action(action)
            reward_t = 9  # end of episode reward
            next_state_t = self.transform_state(next_state)
            done_t = True

        # Otherwise, transform relatively unchanged for sub-problem
        else:
            state_t = self.transform_state(state)
            action_t = self.transform_action(action)
            reward_t = -1
            next_state_t = self.transform_state(next_state)
            done_t = False

        # Pass transformed experience to sub-agent
        self.sub_agent.step(state_t, action_t, reward_t, next_state_t, done_t)
        (self.alpha, self.epsilon, self.gamma) = \
            self.sub_agent.alpha, self.sub_agent.epsilon, self.sub_agent.gamma

    def transform_state(self, state):
        """Transform state into the problem space of the sub-agent"""

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(state)

        # If we don't have the passenger, passenger is our destination
        if pass_idx != self.passenger_in_taxi_idx:
            dest_idx_t = pass_idx
        # If we have the passenger, destination is our destination
        else:
            dest_idx_t = dest_idx

        # Encode in subproblem state space and return
        return (taxi_row, taxi_col, dest_idx_t)

    def transform_action(self, action):
        # Action space is the same, minus the final two actions
        assert action != self.action_pickup
        assert action != self.action_dropoff
        return action

    def can_pick_up(self, state):
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(state)

        # Can't pickup if passenger already in taxi
        if pass_idx == self.passenger_in_taxi_idx:
            return False

        # Otherwise, taxi must be colocated with passenger
        return (taxi_row, taxi_col) == self.locs[pass_idx]

    def can_drop_off(self, state):
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(state)

        # Can't dropoff if passenger not in taxi
        if pass_idx != self.passenger_in_taxi_idx:
            return False

        # Otherwise, taxi must be colocated with destination
        return (taxi_row, taxi_col) == self.locs[dest_idx]

    def decode_state(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        taxi_row, taxi_col, pass_idx, dest_idx = reversed(out)
        return taxi_row, taxi_col, pass_idx, dest_idx
