import tkinter as tk
import numpy as np
# tutorial at: https://www.youtube.com/watch?v=b4XP2IcI-Bg

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
        self.make_GUI()
        self.start_game()

        self.master.bind('<Left>', self.left)
        self.master.bind('<Right>', self.right)
        self.master.bind('<Up>', self.up)
        self.master.bind('<Down>', self.down)

        self.mainloop()
    
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
        score_frame.place(relx=0.5, y=45, anchor='center')
        tk.Label(
            score_frame,
            text='Score',
            font='black'
        ).grid(row=0)

        self.score_label = tk.Label(score_frame, text='0', font=("Arial", 12))
        self.score_label.grid(row=1)

    def start_game(self):
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

        self.score = 0


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
        return ret

    def combine(self):
        ret = False
        # diff_mat = np.hstack((self.matrix[:,:3] - self.matrix[:,1:], np.ones((4,1),dtype=int)))
        # self.matrix[diff_mat==0] *= 2
        # self.matrix[np.roll(diff_mat,1,axis=1)==0] = 0
        # self.score += sum(self.matrix[diff_mat==0])
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] != 0 and self.matrix[i][j] == self.matrix[i][j+1]:
                    self.matrix[i][j] *= 2
                    self.matrix[i][j+1] = 0
                    self.score += self.matrix[i][j]
                    ret = True
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
            game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor='center')
            tk.Label(
                game_over_frame,
                text='you win!'
            ).pack()
        elif not np.any(self.matrix==0) and not self.vertical_move_exists() and not self.horizontal_move_exists():
            game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor='center')
            tk.Label(
                game_over_frame,
                text='game over!'
            ).pack()


def main():
    Game()


if __name__ == '__main__':
    main()