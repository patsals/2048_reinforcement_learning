# RL environment guide at: https://www.gymlibrary.dev/content/environment_creation/
# GUI/game tutorial at: https://www.youtube.com/watch?v=b4XP2IcI-Bg
import tkinter as tk
import numpy as np


class Sim(tk.Frame):

    def __init__(self, argState=None, numStepsAhead=0):
        tk.Frame.__init__(self)

        self.action_to_movement = {0: self.left, 1: self.right, 2: self.up, 3: self.down}
        self.rewards = {'won': 100, 'lost':0}
        self.game_status = None
        self.matrix = argState
        self.state = None

        self.show_plot = True
        self.scores = []
        self.rewards = []
        self.highest_tiles = []

        self.num_moves = 0
        self.prev_highest_tile = 2
        self.ep_reward = 0
        self.plot_flag = False

        if (argState is None):
            self.make_GUI()
            self.reset()
        
        self.score = 0
        self.prev_score = 0
        self.game_status = None

        self.step_counter = 0
        self.num_steps = numStepsAhead
       
        self.plot_flag = True
    
    
   
    def step(self, action):

        movement = self.action_to_movement[action]
        movement(None)

        ##################
        #### REWARDS #####
        ##################
        # an episode is done if can no longer play anymore
        terminated = self.game_over()
        reward = 0

        highest_tile = np.max(self.matrix)
        reward += (self.num_moves-6)*2
        if (highest_tile > self.prev_highest_tile):
            reward += highest_tile

        self.num_moves = 0
        self.prev_highest_tile = highest_tile

        self.ep_reward += reward

        # self.prev_matrix = self.matrix

        return np.array(self.state, dtype=np.float32), reward, terminated, self.score



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
                    self.num_moves += 1
        self.matrix = new_matrix
        self.state = self.matrix.flatten()
        return ret

    def combine(self):
        ret = False
        self.prev_score = self.score
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] != 0 and self.matrix[i][j] == self.matrix[i][j+1]:
                    self.matrix[i][j] *= 2
                    self.matrix[i][j+1] = 0
                    self.score += self.matrix[i][j]
                    ret = True
                    self.num_moves += 1
        self.state = self.matrix.flatten()
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
        self.state = self.matrix.flatten()


    def left(self, event):
        self.stack()
        self.combine()
        self.stack()
        self.add_new_tile()
        # self.update_GUI()
        self.step_counter += 1
        self.game_over()

    def right(self, event):
        self.reverse()
        self.stack()
        self.combine()
        self.stack()
        self.reverse()
        self.add_new_tile()
        # self.update_GUI()
        self.step_counter += 1
        self.game_over()

    def up(self, event):
        self.transpose()
        self.stack()
        self.combine()
        self.stack()
        self.transpose()
        self.add_new_tile()
        # self.update_GUI()
        self.step_counter += 1
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
        # self.update_GUI()
        self.step_counter += 1
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

        if (self.step_counter >= self.num_steps):
            return True

        if np.any(self.matrix==2048):
   
            self.game_status = 'won'
            return True

        elif not np.any(self.matrix==0) and not self.vertical_move_exists() and not self.horizontal_move_exists():

            self.game_status = 'lost'
            return True
        
        return False


    def play(self):
        self.mainloop()

    def terminate(self):
        self.master.destroy()
