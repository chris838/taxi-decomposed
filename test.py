import numpy as np
from collections import deque, defaultdict
import random, sys, math, gym

from drop_pick_agent import DropPickAgent

env = gym.make('Taxi-v2')
state = env.reset()
taxirow, taxicol, passloc, destidx = env.env.decode(state)
print(taxirow, taxicol, passloc, destidx)
