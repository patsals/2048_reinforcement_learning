from RL_game_2048 import Game
from DeepQNetwork import Agent
from util import plot_game_scores

from threading import Thread
import time


if __name__ == '__main__':
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.01, eps_dec=0.01, input_dims=[16], lr=0.03)

    n_games = 500
    game = Game()

    #run_game = lambda : game.play()
    #game_thread = Thread(target=run_game)
    #game_thread.start()

    game_scores = []
    for i in range(n_games):
        score = 0
        done = False
        observation = game.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_new, reward, done, game_score = game.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_new, done)
            agent.learn()
            observation = observation_new

        game_scores.append(game_score)
        if i % 10 == 0:
            print(f'episode {i} | total reward score: {score} | game_score: {game_score}')
            #print(game_score, done)

            #time.sleep(0.1)
    #game_thread.join()
    game.terminate()
    plot_game_scores(game_scores)
