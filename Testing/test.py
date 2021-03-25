import numpy as np
import gym
from utils import *

env = gym.make('FetchReach-v1')

print(env.reset())

testarr = [0,0.05,0,0]
states = []
rewards = []
done = []

for _ in range(100):
    env.render()
    state, reward, done, _ = env.step(testarr)
    rewards.append(reward)
env.close()
print(state['observation'])