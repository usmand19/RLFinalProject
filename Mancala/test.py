from env import MancalaEnv
from game import Game
from randomagent import AgentRandom

import numpy as np

game = Game()
random_agent = AgentRandom()
random_agent2 = AgentRandom()

testarr = np.zeros(shape = (1,2))

testarr = np.append(testarr, [[1,3]], axis = 0)
testarr = np.append(testarr, [[2,4]], axis = 0)
print(testarr)

'''
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