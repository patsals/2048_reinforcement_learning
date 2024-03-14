import matplotlib.pyplot as plt
import numpy as np

def plot_game_scores(scores):
    plt.plot(list(zip(scores)))
    plt.xlabel('episode')
    plt.ylabel('game score')
    plt.title('Game scores over episodes')
    plt.savefig('plots/game_scores.png')