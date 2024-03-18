# RL environment guide at: https://www.gymlibrary.dev/content/environment_creation/
# GUI/game tutorial at: https://www.youtube.com/watch?v=b4XP2IcI-Bg
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import torch.nn.functional as F

class Game(tk.Frame):

    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("2048")

        self.NUM_COLORS = {
            2: '#eee4da',
            4: '#eee1c9',
            8: '#f3b72a',
            16: '#f69664',
            32: '#f77c5f',
            64: '#f75f3b',
            128: '#edd073',
            256: '#edcc62',
            512: '#edc950',
            1024: '#edc53f',
            2048: '#edc22e'
        }

        self.main_grid = tk.Frame(
            self, bg='grey', bd=3, width=600, height=600
        )

        self.main_grid.grid(pady=(100,0))
        

        # self.master.bind('<Left>', self.left)
        # self.master.bind('<Right>', self.right)
        # self.master.bind('<Up>', self.up)
        # self.master.bind('<Down>', self.down)

        self.action_to_movement = {0: self.left, 1: self.right, 2: self.up, 3: self.down}
        self.rewards = {'won': 100, 'lost':0}
        self.game_status = None
        self.matrix = None
        self.state = None

        self.show_plot = True
        self.scores = []
        self.highest_tiles = []

        self.make_GUI()
        self.reset()


    # def get_observation(self):
    #     return {'agent': self.score, 'target':2048}
    
    def reset(self):
        self.matrix = np.zeros((4,4),dtype=int)
        

        # fill 2 random cells
        ind = np.random.choice(range(16),2,replace=False)
        row = ind//4
        col = ind%4
        self.matrix[row[0]][col[0]] = 2
        self.matrix[row[1]][col[1]] = 2
        self.cells[row[0]][col[0]]['frame'].configure(bg=self.get_color(2))
        self.cells[row[0]][col[0]]['number'].configure(
            bg=self.NUM_COLORS[2],
            text='2'
        )
        self.cells[row[1]][col[1]]['frame'].configure(bg=self.get_color(2))
        self.cells[row[1]][col[1]]['number'].configure(
            bg=self.NUM_COLORS[2],
            text='2'
        )

        if len(self.scores) == 0:
            self.scores.append(0)
            self.highest_tiles.append(0)
        elif self.score != 0:
            self.scores.append(self.score)
            self.highest_tiles.append(self.matrix.max())
        
        self.score = 0
        self.prev_score = 0
        self.game_status = None
        #self.state = np.array(self.matrix.flatten(), type=np.float32)
        self.state = self.encode_state(self.matrix)
        #observation = self.get_observation()
        
        self.update_line_plot()
        self.update_histogram_plot()
        return self.state
    
   
    def step(self, action):
        movement = self.action_to_movement[action]
        movement(None)

        ##################
        #### REWARDS #####
        ##################
        # an episode is done if can no longer play anymore
        terminated = self.game_over()
        # if terminated:
        #     reward = self.rewards[self.game_status]
        # else:
        #     reward = 0

        reward = 0

        # highest_tile = np.max(self.matrix)
        # if highest_tile not in [self.matrix[0][0], self.matrix[0][3], self.matrix[3][0], self.matrix[0][3]]:
        #     reward -= 100
        # else:
        #     #reward += self.score - self.prev_score
        #     reward += 100
        
        # higher_tiles = np.array([64, 128, 256, 512])
        # corner_tiles  = np.array([self.matrix[0][0], self.matrix[0][3], self.matrix[3][0], self.matrix[0][3]])
        # if not np.isin(higher_tiles, corner_tiles).any():
        #     reward -= 100
        reward += self.score - self.prev_score
        #state = np.array(self.state, dtype=np.float32)
        self.state = self.encode_state(self.matrix)
        return self.state, reward, terminated, self.score


    # FOR CNN IMPLEMENTATION
    def encode_state(self, board):
        board_flat = [0 if e == 0 else int(np.log2(e)) for e in board.flatten()]
        board_flat = torch.LongTensor(board_flat)
        board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
        board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
        return board_flat

    def get_color(self, val):
        return self.NUM_COLORS[val] if val in self.NUM_COLORS else 'tan'

    def make_GUI(self):
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid,
                    bg='white',
                    width=150,
                    height=150
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg='white', font=("Arial bold", 40))
                cell_number.grid(row=i, column=j)
                cell_data = {'frame': cell_frame, 'number': cell_number}
                row.append(cell_data)
            self.cells.append(row)

        score_frame = tk.Frame(self)
        score_frame.place(relx=0.25, y=45, anchor='center')
        tk.Label(
            score_frame,
            text='Score',
            font=("Arial bold", 40)
        ).grid(row=0)

        self.score_label = tk.Label(score_frame, text='0', font=("Arial bold", 20))
        self.score_label.grid(row=1)

        if self.show_plot:
            self.fig, self.ax = plt.subplots(2, figsize=(5, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.grid(row=0, column=5, rowspan=4, padx=20, pady=20)

            self.update_line_plot()
            self.update_histogram_plot()
            

    def update_line_plot(self):
        self.ax[0].clear()
        self.ax[0].plot(self.scores)
        self.ax[0].set_title('Game scores over episodes', fontsize=10)
        #self.ax[0].set_xlabel('episode', fontsize=5)
        self.ax[0].set_ylabel('game score', fontsize=8)
        self.ax[0].tick_params(axis='both', labelsize=6)
        self.canvas.draw()

    def update_histogram_plot(self):
        self.ax[1].clear()
        unique_tiles, counts = np.unique(self.highest_tiles, return_counts=True)
        bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        counts_dict = dict(zip(unique_tiles,counts))
        self.ax[1].bar(np.array(bins).astype(str), [counts_dict.get(tile, 0) for tile in bins], color='blue')
        self.ax[1].set_title('Highest Tile Distribution', fontsize=10)
        self.ax[1].set_xlabel('Highest Tile', fontsize=8)
        self.ax[1].set_ylabel('Count', fontsize=8)
        self.ax[1].tick_params(axis='both', labelsize=6)

        self.canvas.draw()



    def stack(self):
        ret = False
        new_matrix = np.zeros((4,4),dtype=int)
        for i in range(4):
            fill_position = 0
            for j in range(4):
                if self.matrix[i][j] != 0:
                    new_matrix[i][fill_position] = self.matrix[i][j]
                    fill_position += 1
                    ret = True
        self.matrix = new_matrix
        #self.state = self.matrix.flatten()
        self.state = self.encode_state(self.matrix)
        return ret

    def combine(self):
        ret = False
        # diff_mat = np.hstack((self.matrix[:,:3] - self.matrix[:,1:], np.ones((4,1),dtype=int)))
        # self.matrix[diff_mat==0] *= 2
        # self.matrix[np.roll(diff_mat,1,axis=1)==0] = 0
        # self.score += sum(self.matrix[diff_mat==0])
        self.prev_score = self.score
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] != 0 and self.matrix[i][j] == self.matrix[i][j+1]:
                    self.matrix[i][j] *= 2
                    self.matrix[i][j+1] = 0
                    self.score += self.matrix[i][j]
                    ret = True
        #self.state = self.matrix.flatten()
        self.state = self.encode_state(self.matrix)
        return ret

    def reverse(self):
        self.matrix = np.flip(self.matrix, axis=1)

    def transpose(self):
        self.matrix = self.matrix.T

    def add_new_tile(self):
        ind = np.arange(16,dtype=int)[np.ndarray.flatten(self.matrix) == 0]
        if (len(ind) > 0):
            ind = np.random.choice(ind,1)[0]
            row = ind//4
            col = ind%4
            self.matrix[row][col] = np.random.choice([2,4],p=[0.9,0.1])
        #self.state = self.matrix.flatten()
        self.state = self.encode_state(self.matrix)

    def update_GUI(self):
        for i in range(4):
            for j in range(4):
                cell_value = self.matrix[i][j]
                if cell_value == 0:
                    self.cells[i][j]['frame'].configure(bg='white')
                    self.cells[i][j]['number'].configure(bg='white', text='')
                else:
                    color = self.get_color(self.matrix[i][j])
                    self.cells[i][j]['frame'].configure(bg=color)
                    self.cells[i][j]['number'].configure(bg=color, text=str(cell_value))

        self.score_label.configure(text=self.score)
        self.update_idletasks()

    def left(self, event):
        self.stack()
        self.combine()
        self.stack()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def right(self, event):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def up(self, event):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()

    def down(self, event):
        self.transpose()
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.transpose()
        self.add_new_tile()
        self.update_GUI()
        self.game_over()


    # check if any moves are possible
    def horizontal_move_exists(self):
        diff_mat = self.matrix[:,:3] - self.matrix[:,1:]
        return np.any(diff_mat==0)
    
    def vertical_move_exists(self):
        diff_mat = self.matrix[:3,:] - self.matrix[1:,:]
        return np.any(diff_mat==0)
    
    # check if game is over win/lose
    def game_over(self):
        if np.any(self.matrix==2048):
            # game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            # game_over_frame.place(relx=0.5, rely=0.5, anchor='center')
            # tk.Label(
            #     game_over_frame,
            #     text='you win!'
            # ).pack()
            #print(f'game over| WON with score of {self.score}')
            self.game_status = 'won'
            return True

        elif not np.any(self.matrix==0) and not self.vertical_move_exists() and not self.horizontal_move_exists():
            # game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            # game_over_frame.place(relx=0.5, rely=0.5, anchor='center')
            # tk.Label(
            #     game_over_frame,
            #     text='game over!'
            # ).pack()
            #print(f'game over| LOST with score of {self.score}')
            self.game_status = 'lost'
            return True
        
        return False


    def play(self):
        self.mainloop()

    def terminate(self):
        self.master.destroy()
