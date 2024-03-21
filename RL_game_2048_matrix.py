# RL environment guide at: https://www.gymlibrary.dev/content/environment_creation/
# GUI/game tutorial at: https://www.youtube.com/watch?v=b4XP2IcI-Bg
import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F

class Game():
    def __init__(self, show_board=True):
        self.action_to_movement = {0: self.left, 1: self.right, 2: self.up, 3: self.down}
        self.rewards = {'won': 100, 'lost':0}
        self.game_status = None
        self.matrix = np.zeros((4,4),dtype=int)
        self.state = None

        self.show_plot = True
        self.rewards = []
        self.show_board = show_board
        self.game_number = 0
        self.reset()


    
    def reset(self):
        if self.show_board and self.game_number > 0:
            print(f'game {self.game_number} | score: {self.score} , highest_tile: {self.matrix.max()}')
            print(self.matrix)

        #self.highest_tiles.append(self.matrix.max())
        self.matrix = np.zeros((4,4),dtype=int)
        

        # fill 2 random cells
        ind = np.random.choice(range(16),2,replace=False)
        row = ind//4
        col = ind%4
        self.matrix[row[0]][col[0]] = 2
        self.matrix[row[1]][col[1]] = 2


        # if len(self.scores) == 0:
        #     self.scores.append(0)
        # elif self.score != 0:
        #     self.scores.append(self.score)
            
        
        self.score = 0
        self.prev_score = 0
        self.game_status = None


        self.state = np.array(self.state, dtype=np.float32) # for Linear
        #self.state = self.encode_state(self.matrix) # for CNN
        
        #self.update_line_plot()
        #self.update_histogram_plot()
        self.game_number += 1
        return self.state
    
   
    def step(self, action):
        movement = self.action_to_movement[action]
        movement(None)

        ##################
        #### REWARDS #####
        ##################
        # an episode is done if can no longer play anymore
        terminated = self.game_over()

        reward = 0

        reward += self.score - self.prev_score

        self.state = np.array(self.state, dtype=np.float32) # for Linear

        #self.state = self.encode_state(self.matrix) # for CNN
        return self.state, reward, terminated, self.score, self.matrix.max()


    # FOR CNN IMPLEMENTATION
    def encode_state(self, board):
        board_flat = [0 if e == 0 else int(np.log2(e)) for e in board.flatten()]
        board_flat = torch.LongTensor(board_flat)
        board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()
        board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
        return board_flat



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
        self.state = self.matrix.flatten() # for Linear

        #self.state = self.encode_state(self.matrix) # for CNN
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
        self.state = self.matrix.flatten() # for Linear

        #self.state = self.encode_state(self.matrix) # for CNN
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
        self.state = self.matrix.flatten() # for Linear

        #self.state = self.encode_state(self.matrix) # for CNN


    def left(self, event):
        self.stack()
        self.combine()
        self.stack()
        self.add_new_tile()
        self.game_over()

    def right(self, event):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.add_new_tile()
        self.game_over()

    def up(self, event):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        self.add_new_tile()
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
            self.game_status = 'won'
            return True
        elif not np.any(self.matrix==0) and not self.vertical_move_exists() and not self.horizontal_move_exists():
            self.game_status = 'lost'
            return True
        
        return False




