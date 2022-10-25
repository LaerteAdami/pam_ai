class GameGrid():
    """
    Defines the object game_grid which stores all the topological
    information of the grid and allows to create the grid with a random method

    """

    def __init__(self, height, width):
        self.h = height
        self.w = width
  

    def generate_grid(self):

        import numpy as np
        import random

        random_grid = np.zeros((self.h,self.w),dtype=int)
        for i in range(self.h):
            for j in range(self.w):
                random_grid[i][j] = random.randrange(9)
        self.grid = random_grid

    def print_grid(self):

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()

        for i in range(self.grid.shape[0]-1,-1,-1):
            for j in range(self.grid.shape[1]):
                c = self.grid[i][j]    
                ax.text(j+0.5, self.grid.shape[0]-i-0.5, str(c), va='center', ha='center')


        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.set_xticks(np.arange(self.grid.shape[1]+1))
        ax.set_yticks(np.arange(self.grid.shape[0]+1))
        plt.xticks(color='w') 
        plt.yticks(color='w') 
        ax.grid()     


        import matplotlib.ticker as ticker

        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.grid()

        # plt.matshow(Test,cmap=plt.cm.Reds,alpha = 0.5)    