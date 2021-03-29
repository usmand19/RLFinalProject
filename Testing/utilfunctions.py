# utilfunctions.py

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

from global_vars import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def parse_state(state):
    '''
    Given the state observation of the environment,
    extract and return:
    qpos: the new state (q-positions) of the robot
    a_goal: achieved goal of the robot
    d_goal: deisred goal of the robot
    is_done: Whether or not the state was terminal
    '''
    q_pos = state['observation']
    a_goal = state['achieved_goal']
    d_goal = state['desired_goal']
    return(q_pos, a_goal, d_goal)

def to_tensor(arr):
    return_val = np.reshape(arr, (1, -1))
    return_val = torch.tensor(return_val).float()
    return return_val

def return_action_list(index):
    '''
    Given an index of 8 actions, return a list to be used as an environment action
    where the speed at the joint is defined by speed_param
    '''
    return_speeds = [0,0,0,0]
    global speed_param
    speed_param = 1
    if(index % 2) == 0:
        multiplier = 1
    else:
        multiplier = -1
    return_speeds[int(index/2)] = multiplier * speed_param
    return return_speeds

    
def select_action(state):


    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            index = policy_net(state).max(1)[1].view(1, 1)
            return steps_done, index
    else:
        index = torch.tensor( [[float(random.randint(0,n_actions-1))]], device=device, dtype=torch.long)
        #index = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return steps_done, index

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
    # if the number of memories in the buffer is not enough for a batch,
    # do nothing and return
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()




