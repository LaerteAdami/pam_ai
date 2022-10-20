class GameGrid():
    """
    Defines the object game_grid which stores all the topological
    information of the grid and allows to create the grid with a random method

    """

    def __init__(self, height = int, width = int):
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