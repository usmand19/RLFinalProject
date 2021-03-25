import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import tqdm as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class ReplayBuffer(object):
    '''
    Replay Buffer used in DQN. Has parameters
    memory: named tuples contained in the buffer
    Transition: named tuple container for the tuples
    capacity: max size of memory buffer
    position: 

    And methods:
    push: Add an element to the memory buffer. If buffer is full, overwrites the buffer starting from oldest
    sample: return a batch of tuples from the memory
    __len__: get current number of memories stored in buffer
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []            
        self.position = 0
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent():
    def __init__(self, h, w, outputs):
        super().__init__()
        # Inputs to hidden layer
        self.hidden1 = nn.Linear(10,12)
        self.output = nn.Linear(12,4)

        # Activation Functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return (x)


def get_qpos(state):
    '''
    Given the state observation of the environment,
    extract and return the joint (q) positions of the robot
    '''
    return(state['observation'])
    
def get_achieved_reward(state):
    '''
    Given the state observation of the environment,
    extract and return the joint (q) positions of the robot
    '''
    return(state['observation'])

def get_desired_reward(state):
    '''
    Given the state observation of the environment,
    extract and return the joint (q) positions of the robot
    '''
    return(state['observation'])

