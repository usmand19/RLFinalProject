import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from tqdm import tqdm

from utils import *
from utilfunctions import *
from global_vars import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('FetchReach-v1').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 10
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_state = env.reset()
init_q, _, _ = parse_state(init_state)

n_actions = 8

policy_net = DQNAgent(n_actions).to(device)
target_net = DQNAgent(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000, Transition)
steps_done = 0
episode_durations = []


num_episodes = 10
STEP_LIMIT = 500
speed_param = 1

t = 0
done = False
env.render()
for i_episode in tqdm(range(num_episodes)):
    # Initialize environment and state
    state, _, _= parse_state(env.reset())
    state = to_tensor(state)
    while not done or t < STEP_LIMIT:
        # Select and perform an action
        steps_done, action = select_action(state, EPS_START, EPS_END, EPS_DECAY, steps_done, policy_net, n_actions, device)
        obs, reward, done, _ = env.step(return_action_list(action))
        env.render()
        reward = torch.tensor([reward], device=device)
        

        # Observe new state
        if not done:
            next_state, _, _ = parse_state(obs)
            next_state = to_tensor(next_state)
        else:
            next_state = None
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        #perform one step of optimization (on the target network)
        optimize_model(memory, BATCH_SIZE, policy_net, target_net, Transition, device)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        t += + 1
    # Update the target network every TARGET_UPDATE, copying all weights and biases
    # in our policy DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
env.render()
print('Complete')
env.close()
