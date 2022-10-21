class DijkstraSearch():

    def __init__(self, agent, random_grid):
        self.agent = agent
        self.grid = random_grid.grid
        self.w = random_grid.grid.shape[1]
        self.h = random_grid.grid.shape[0]
        self.path = ""

    def compute_path(self):
        """
        This method should decide in which direction to move by evaluating the cell to the right 
        and below the selected cell (excluding special cases at the edges)
        For this cell, we evaluate the sum of the cell + the subsequent cells below and to the right
        The direction chosen is the one with the smallest sum
        In this way we can evaluate with a little more depth the best direction for each step
        """

        h = self.h - 1
        w = self.w - 1
        path = []

        self.agent.timer = self.grid[0,0] # initialise the timer of the agent
        i = self.agent.i # horizontal position of the agent at the start
        j = self.agent.j # vertical position of the agent at the start

        while not((i==h) and (j==w)):
        
            if j == w: # only move down
                move = "D"
                self.agent.timer += self.grid[i+1,j]
                i+=1
                path.append(move)
                continue
            
            if i == h: # only move right
                move = "R"
                self.agent.timer += self.grid[i,j+1]
                j+=1
                path.append(move)
                continue
            
            # right cell
            time_right = self.compute_time(i+1, j+1) + self.compute_time(i, j+1) + self.compute_time(i, j+2)

            # down cell
            time_down = self.compute_time(i+1, j) + self.compute_time(i+2, j)+ self.compute_time(i+1, j+1)

            if time_right < time_down:
                move = "R"
                self.agent.timer += self.grid[i,j+1]
                j+=1
            else:
                move = "D"
                self.agent.timer += self.grid[i+1,j]
                i+=1

            path.append(move)



        self.agent.path = (" ".join(path))      

    def compute_time(self, i,j):
        if (j > self.w - 1  or i > self.h -1 ):
            return 0 
        return self.grid[i,j]     








