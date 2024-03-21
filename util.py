import matplotlib.pyplot as plt
import numpy as np

def plot_game_scores(scores):
    plt.clf()
    plt.plot(list(zip(scores)))
    plt.xlabel('episode')
    plt.ylabel('game score')
    plt.title('Game scores over episodes')
    plt.savefig('plots/game_scores.png')

def plot_highest_tiles(tiles):
    plt.clf()
    unique_tiles, counts = np.unique(tiles, return_counts=True)
    bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    counts_dict = dict(zip(unique_tiles,counts))
    plt.bar(np.array(bins).astype(str), [counts_dict.get(tile, 0) for tile in bins], color='blue')
    plt.title('Highest Tile Distribution', fontsize=10)
    plt.xlabel('Highest Tile', fontsize=8)
    plt.ylabel('Count', fontsize=8)
    plt.tick_params(axis='both', labelsize=6)
    plt.savefig('plots/highest_tiles.png')
