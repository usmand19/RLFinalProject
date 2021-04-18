from game import Game
from randomagent import AgentRandom
from exactagent import AgentExact
from maxagent import AgentMax

import random
import numpy as np

class MancalaEnv():

    def __init__(self, reward_type = "sparse_reward", opponent = 1, debug = False):
        self.game = Game()
        self.reward_type = reward_type
        self.score = (0,0)
        self.debug = debug
        self.action_space = 6
        self.observation_space = 14

        if opponent == 1:
            self.opponent = AgentRandom()
        if opponent == 2:
            self.opponent = AgentExact()
        if opponent == 3:
            self.opponent = AgentMax()

    def opponent_move(self, player_id):
        
        if player_id == 1:
            self.game._player_one = False
        else:
            self.game._player_one = True
        
        # Agent Takes Action
        action = self.opponent._move(self.game)
        if self.debug:
            print(f"Board:\n{self.game.board()}\nOpponent ID: {self.game._player_one}\tOpponent Action: {action}")
        self.game.move(action)
        if self.debug:
            print(f"New Board:\n{self.game.board()}\n")

        
    def reset(self):
        '''
        Reset the game board back to original, and return starting state
        '''
        self.game = Game()
        self.score = (0,0)
        state = np.array(self.game.board())
        return(state)
    
    '''
    ##########  REWARD FUNCTIONS ##########
    Given an action, computes:
        * the next state
        * the reward as specified
        * Which player has won the game (0,1,2) where 0 is no player (yet)
        * if the game is done (a player has won)
    '''

    def sparse_reward(self, action, player_id):

        
        # Legalize action:
        if player_id == 1:
            self.game._player_one = True
        else:
            action += 7
            self.game._player_one = False
        if self.debug:
            print(f"Board:\n{self.game.board()}\nAgent ID: {self.game._player_one}\tAgent Action: {action}")

        # Take the action
        self.score = self.game.move(action)
        
        # Obtain the next state
        next_state = np.array(self.game.board())
        #Opponent Move
        self.opponent_move(1)
        # Get if game is finished
        done = self.game.over()
        # Get Winner of game:
        winner = self.score.index(max(self.score)) + 1
        # If finished....
        if done:
            # And agent won
            if (winner == player_id):
                reward = 10
                win = True
            # And agent lost
            else:
                reward = -10
                win = False
        else:
            reward = 0
            win = False
        if self.debug:
            print(f"New Board:\n{self.game.board()}\nReward: {reward}\tScore: {self.get_score()}\n")
        return(next_state, reward, done, win)

    def keep_ahead_reward(self, action, player_id):

        # Legalize action:
        if player_id == 1:
            self.game._player_one = True
        else:
            action += 7
            self.game._player_one = False


        # Take the action

        self.score = self.game.move(action)
        
        next_state = np.array(self.game.board())

        self.opponent_move(1)
        # Get if game is finished
        done = self.game.over()
        # Get Winner of game:
        winner = self.score.index(max(self.score)) + 1
        # Get if agent is ahead or not
        ahead = self.score[0] - self.score[1]
        # If finished....
        if done:
            # And agent won
            if (winner == player_id):
                reward = 10
                win = True
            # And agent lost
            else:
                reward = -10
                win = False
        else:
            if ahead > 0:
                reward = 0.1
            else:
                reward = 0
            win = False
        return(next_state, reward, done, win)

    def dont_fall_behind_reward(self, action, player_id):
            
        # Legalize action:
        if player_id == 1:
            self.game._player_one = True
        else:
            action += 7
            self.game._player_one = False

        # Take the action

        self.score = self.game.move(action)
        self.opponent_move(1)
        # Obtain the next state
        next_state = np.array(self.game.board())
        # Get if game is finished
        done = self.game.over()
        # Get Winner of game:
        winner = self.score.index(max(self.score)) + 1
        # Get if agent is ahead or not
        ahead = self.score[0] - self.score[1]
        # If finished....
        if done:
            # And agent won
            if (winner == player_id):
                reward = 10
                win = True
            # And agent lost
            else:
                reward = -10
                win = False
        else:
            if ahead < 0:
                reward = -1
            else:
                reward = 0
            win = False
        return(next_state, reward, done, win)

    def step(self, action, player_id):


        if self.reward_type == "sparse_reward":
            results = self.sparse_reward(action, player_id)

        if self.reward_type == "keep_ahead_reward":
            results = self.keep_ahead_reward(action, player_id)

        if self.reward_type == "dont_fall_behind_reward":
            results = self.dont_fall_behind_reward(action, player_id)


        return(results)

    def sample(self, player_id):
        if player_id == 1:
            moves = [0,1,2,3,4,5]
            return random.sample(moves, 1)[0]
        if player_id == 2:
            moves = [7,8,9,10,11,12]
            return random.sample(moves, 1)[0]


    def get_score(self):
        score = self.game.score()
        return(score)

    def set_player(self, player_id):
                # Legalize action:
        if player_id == 1:
            self.game._player_one = True
        else:
            self.game._player_one = False
        return(self.game._player_one)
