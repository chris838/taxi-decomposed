import numpy as np
from collections import deque, defaultdict
import random, sys, math, gym

from decomp_agent import DecompAgent

def train_episode(env, agent):
    state = env.reset()
    episode_return = 0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        episode_return += reward
        state = next_state
        if done:
            return episode_return

env = gym.make('Taxi-v2')
agent = DecompAgent()

episode_i = 0
num_episodes = 100
window = 100
episode_returns = deque(maxlen=window)
best_sample_avg = -np.inf

for i in range(num_episodes):

    episode_return = train_episode(env, agent)
    episode_returns.append(episode_return)

    # best 100 sample average
    if len(episode_returns) >= window:
        sample_average = np.mean(episode_returns)
        best_sample_avg = max(best_sample_avg, np.mean(episode_returns))
    #
    episode_i += 1

print(best_sample_avg)
