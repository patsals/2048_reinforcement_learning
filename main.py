from RL_game_2048 import Game
from DeepQNetwork import Agent





if __name__ == '__main__':
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                eps_end=0.01, input_dims=[4], lr=0.003)

    n_games = 10
    game = Game()#.play()

    for i in range(n_games):
        score = 0
        done = False
        observation = game.reset()

        while not done:
            print(1)
            action = agent.choose_action(observation)
            observation_new, reward, done = game.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_new, done)
            agent.learn()
            observation = observation_new
        print(score)
