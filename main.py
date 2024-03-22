from RL_game_2048 import Game_GUI
from RL_game_2048_matrix import Game
from DeepQNetwork import Agent, DeepQNetworkLinear
from util import plot_game_scores, plot_highest_tiles
import torch

from threading import Thread
import random

game_scores = []
highest_tiles = []
SHOW_GUI = False

def train(game, agent):
    for i in range(n_games):
        score = 0
        done = False
        observation = game.reset()
        while not done:
            action = random.randint(0,3)#agent.choose_action(observation)
            observation_new, reward, done, game_score, highest_tile = game.step(action)
            score += reward
            #agent.store_transition(observation, action, reward, observation_new, done)
            #agent.learn()
            observation = observation_new
        game_scores.append(game_score)
        highest_tiles.append(highest_tile)
        if i % 10 == 0:
            print(f'episode {i} | total reward score: {score} | average game_score: {sum(game_scores[-10:])/10}')
            plot_game_scores(game_scores)
            plot_highest_tiles(highest_tiles)
            
        if len(game_scores) > 1 and game_score > max(game_scores[:-1]):
            torch.save(agent.Q_eval.state_dict(), 'best_linear_model.pth')

        
if __name__ == '__main__':

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.01, input_dims=[16], lr=0.01)

    n_games = 100_000
    
    if SHOW_GUI:
        game = Game_GUI()
        train_thread = Thread(target=lambda : train(game, agent))
        train_thread.start()
        game.play()
        train_thread.join()
        game.terminate()
    else:
        game = Game(show_board=False)
        train(game, agent)
        
    
    model = DeepQNetworkLinear(n_actions=4, input_dims=[16], lr=0.01, fc1_dims=256, fc2_dims=256)
    model.load_state_dict(torch.load('best_linear_model.pth'))
    model.eval()  
    
