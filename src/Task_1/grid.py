import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class GameGrid:
    """
    Defines the object game_grid which stores all the topological
    information of the grid and allows to create the grid with a random method

    """

    def __init__(self, height, width):
        self.h = height
        self.w = width

    def generate_grid(self):

        # generate the grid with random values in [0,9]
        self.grid = np.random.randint(0, 10, size=(self.h, self.w), dtype=int)

    def print_grid(self, save_path=""):

        fig, ax = plt.subplots()

        for i in range(self.grid.shape[0] - 1, -1, -1):
            for j in range(self.grid.shape[1]):
                c = self.grid[i][j]
                ax.text(
                    j + 0.5,
                    self.grid.shape[0] - i - 0.5,
                    str(c),
                    va="center",
                    ha="center",
                )

        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.set_xticks(np.arange(self.grid.shape[1] + 1))
        ax.set_yticks(np.arange(self.grid.shape[0] + 1))
        plt.xticks(color="w")
        plt.yticks(color="w")

        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.grid()

        if not (not save_path):  # if save_path is not empty
            plt.savefig(save_path)

        return fig, ax

    def print_path(self, visited_cells, save_path=""):

        fig, ax = self.print_grid()

        for i in range(self.grid.shape[0] - 1, -1, -1):
            for j in range(self.grid.shape[1]):
                if (i, j) in visited_cells:
                    c = self.grid[i][j]
                    ax.text(
                        j + 0.5,
                        self.grid.shape[0] - i - 0.5,
                        str(c),
                        va="center",
                        ha="center",
                        bbox=dict(
                            boxstyle="round",
                            ec=(1.0, 0.5, 0.5),
                            fc=(1.0, 0.8, 0.8),
                        ),
                    )

        if not (not save_path):  # if save_path is not empty
            plt.savefig(save_path)
