import gym
import copy
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import math
import torchvision.transforms as T
import numpy as np
import matplotlib.patches as mpatches
import time
import torch.nn as nn
import itertools

from tqdm import tqdm
from lib.a3c import AgentA3C
from lib.MancalaEnv import MancalaEnv
from lib.game import Game
from lib.randomagent import AgentRandom
from lib.exactagent import AgentExact
from lib.maxagent import AgentMax
from lib.max_min import AgentMinMax

import warnings
warnings.filterwarnings("ignore")

class DQN_replay(nn.Module):
    '''
    This function builds on the DQN class to include the use 
    of a replay memory, where transitions are stored, and randomly
    sampled when the network is updated
    '''
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        super(DQN_replay, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(hidden_dim*2, action_dim)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)



    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQN. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def _move(self, game):
        state = game.board()
        state2 = state
        if (game._player_one == False):
            state2[7:14] = state[0:7]
            state2[0:7] = state[7:14]
            with torch.no_grad():
                q_values = self.model(torch.Tensor(state2))
            action = torch.argmax(q_values).item()
            return(action+7)
        else:
            with torch.no_grad():
                q_values = self.model(torch.Tensor(state))
            action = torch.argmax(q_values).item()    
            return(action)
        
        

    def replay(self, memory, size, gamma=0.9):
        """New replay function"""
        #Try to improve replay speed
        if len(memory)>=size:
            batch = random.sample(memory,size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states)
            #Update q values
            all_q_values[range(len(all_q_values)),actions]=rewards+gamma*torch.max(all_q_values_next, axis=1).values
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()]=rewards[is_dones_indices.tolist()]
        
            
            self.update(states.tolist(), all_q_values.tolist())


def render(state):
    '''
    This function renders the game board
    '''
    mancala1 = state[6]
    mancala2 = state[13]
    dqn = state[0:6]
    player = list(reversed(state[7:13]))
    print(f"DQN:     [{mancala1}]  {dqn}\nPlayer:       {player}  [{mancala2}]")


PATH_DQN = "Replay_dqn_state_dict.pt"
DQN_model = DQN_replay(14,6)
DQN_model.model.load_state_dict(torch.load(PATH_DQN))

game = Game()
done = False

while not done:
    
    # Player 1 Move
    board1 = game.board()
    game._player_one = True
    p1_action = DQN_model._move(game)
    print(f"DQN Action: {p1_action + 1}")
    game.move(p1_action)
    # End game if move meets win condition
    if game.over():
        break

    # Player 2 Move
    render(game.board())
    game._player_one = False
    p2_action = input("Enter an Action between 1 and 6: ")
    p2_action = 13 - (int(p2_action))
    print(p2_action)
    game.move(p2_action)
    render(game.board())

    # End game if move meets win condition
    if game.over():
        break

    # Store game completion flag
    done = game.over()

winner = game.winner()

if winner == 1:
    print("THE HARDEST CHOICES REQUIRE THE STRONGEST WILLS. DQN IS UNBEATABLE!")

if winner == 2:
    print("Fine, you win")
      
