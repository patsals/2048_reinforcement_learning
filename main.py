from RL_game_2048 import Game
from DeepQNetwork import Agent
from util import plot_game_scores
import torch
import os
from threading import Thread
import numpy as np
from sim_2048 import Sim

os.environ['KMP_DUPLICATE_LIB_OK']='True'
game_scores = []

def train(game):
    for i in range(n_games):
        score = 0
        no_motion_ctr = 0
        done = False
        observation = game.reset()
        while not done:
            greedy_action, min_action = simulate(observation, 5, 5)
            
            if (no_motion_ctr >= 10): # 10 is a random var i chose
                action = min_action
                no_motion_ctr = 0
            else:
                action = greedy_action

            observation_new, reward, done, game_score = game.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_new, done)
            agent.learn()
            # agent.update_epsilon(game.prev_score, game.score)
            observation = observation_new

            if (game.prev_score == game.score):
                no_motion_ctr += 1

        game_scores.append(game_score)
        if i % 10 == 0:
            print(f'episode {i} | total reward score: {score} | game_score: {game_score}')

        # torch.save(agent.Q_eval, 'latest_model.pt')


def simulate(observation, num_simulations, num_steps_ahead):
    
    sim_move_scores = np.zeros(4)

    for move in range(4):
        for i in range(num_simulations):
                
                game_copy = Sim(np.reshape(observation, (4, 4)), num_steps_ahead)
                '''
                First step per simulation is in any of the 4 directions

                0: left
                1: right
                2: up
                3: down
                '''
                action_sim = move
                observation_new_sim, reward_sim, done_sim, game_score_sim = game_copy.step(action_sim)
                sim_move_scores[move] += reward_sim
                #agent.store_transition(observation, action_sim, reward_sim, observation_new_sim, done_sim)
                observation_sim = observation_new_sim

                while game_copy.game_over() == False:
                    action = agent.choose_action(observation_sim)
                    observation_new_sim, reward_sim, done_sim, game_score_sim = game_copy.step(action)
                    sim_move_scores[move] += reward_sim
                    #agent.store_transition(observation, action_sim, reward_sim, observation_new_sim, done_sim)
                    observation_sim = observation_new_sim

    sim_move_scores = sim_move_scores / num_simulations

    # print(sim_move_scores)

    return(sim_move_scores.argmax(axis=0), sim_move_scores.argmin(axis=0))
    
if __name__ == '__main__':
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_min=0.01, eps_max=0.20, eps_chg=0.01, input_dims=[16], lr=0.03, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    n_games = 5000
    game = Game()

    train_thread = Thread(target=lambda : train(game))
    train_thread.start()
    game.play()


    train_thread.join()
    game.terminate()
    plot_game_scores(game_scores)
    
