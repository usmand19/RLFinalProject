import numpy as np
import gym
from utils import *
import time

env = gym.make('FetchReach-v1')

env.reset()

testarr = [0,0.05,0,0]
testarr2 = [1,0.05,0,0]
states = []
rewards = []
done = []

for _ in range(100):
    env.render()
    state, reward, done, _ = env.step(testarr)
    rewards.append(reward)
env.close()

env.render()
print(env.step(testarr))
print(env.step(testarr2))
