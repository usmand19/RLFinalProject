"""Agent with uses A3C trained network"""

import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from gym.utils import seeding

from game import Game
from env import MancalaEnv
from baseagent import Agent
from a3c_model import ActorCritic


class AgentA3C(Agent):
    '''Agent which leverages Actor Critic Learning'''

    def __init__(self,
                 model_path,
                 dtype,
                 seed=451):
        self._seed = seed
        self._idx = 0
        self.np_random, _ = seeding.np_random(seed)
        self._dtype = dtype
        self.env = MancalaEnv(seed)
        state = self.env._reset()

        self._model = ActorCritic(
            state.shape[0], self.env.action_space).type(dtype)
        self._model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 

    def _move(self, game):
        '''Return move which ends in score hole'''
        assert not game.over()
        self._idx += 1
        game_clone, rot_flag = game.clone_turn()
        move_options = Agent.valid_indices(game_clone)

        state = self.env.force(game_clone)
        state = torch.from_numpy(state).type(self._dtype)
        cx = Variable(torch.zeros(1, 400).type(self._dtype), volatile=True)
        hx = Variable(torch.zeros(1, 400).type(self._dtype), volatile=True)

        _, logit, (hx, cx) = self._model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        scores = [(action, score) for action, score in enumerate(
            prob[0].data.tolist()) if action in move_options]

        valid_actions = [action for action, _ in scores]
        valid_scores = np.array([score for _, score in scores])

        final_move = self.np_random.choice(valid_actions, 1, p=valid_scores/valid_scores.sum())[0]

        return Game.rotate_board(rot_flag, final_move)
