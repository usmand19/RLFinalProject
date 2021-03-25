import numpy as np
import gym
from utils import *

env = gym.make('FetchReach-v1')
state = env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


print(get_qpos(state))