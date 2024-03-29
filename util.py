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
    counts_dict = dict(zip(unique_tiles, counts))
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'lightblue', 'grey', 'black']
    if len(colors) < len(bins):
        colors += colors * (len(bins) // len(colors)) + colors[:len(bins) % len(colors)]
    
    percentages = [counts_dict.get(tile, 0)/len(tiles) * 100 for tile in bins]

    bars = plt.bar(np.array(bins).astype(str), percentages, color=colors)
    plt.title('Highest Tile Distribution', fontsize=10)
    plt.xlabel('Highest Tile', fontsize=8)
    plt.ylabel('Percentage', fontsize=8)
    plt.xticks(rotation=45)  
    plt.tick_params(axis='both', labelsize=6)

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', ha='center', va='bottom', fontsize=6)

    plt.savefig('plots/highest_tiles.png')
