from env import MancalaEnv
from game import Game
from randomagent import AgentRandom
from exactagent import AgentExact
import numpy as np


game = Game()
print(game._player_one)
game.move(0)
print(game._player_one)
print(game.board())
'''
random_agent = AgentRandom()
random_agent2 = AgentExact()
action = random_agent2._move(game)
print(action)

print(game.board())
game._player_one = True
move = random_agent.move(game)
game.move(move)
print(game.board())


while True:
    print(f"Player {game.turn_player()}'s turn")
    print(game.board())
    # act = input("Action?")
    score = np.asarray(game.move(random_agent2._move(game)))
    print(game.board())
    print(score)
    game._player_one = False
    print(f"Player {game.turn_player()}'s turn")
    print(game.board())
    score = np.asarray(game.move(random_agent._move(game)))
    print(game.board())
    print(f"{score}\n")
    over = game.over()
    if over:
        winner = np.argmax(score) + 1
        print(f"Finished with \n done: {over}\n Winner: {winner}")
        break

'''