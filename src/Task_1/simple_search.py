class SimpleSearch():

    def __init__(self, agent, random_grid):
        self.agent = agent
        self.grid = random_grid.grid
        self.w = random_grid.grid.shape[1]
        self.h = random_grid.grid.shape[0]

    def compute_path(self):
        """
        This method should decide in which direction to move by evaluating the cell to the right 
        and below the selected cell (excluding special cases at the edges)
        For this cell, we evaluate the sum of the cell + the subsequent cells below and to the right
        The direction chosen is the one with the smallest sum
        In this way we can evaluate with a little more depth the best direction for each step
        """
        import time

        start = time.time() # variable used to store the initial time. 
                            # It will be use to compute the total execution time of the search
        
        h = self.h - 1
        w = self.w - 1

        self.agent.timer = self.grid[0,0] # initialise the timer of the agent
        i = self.agent.i # horizontal position of the agent at the start
        j = self.agent.j # vertical position of the agent at the start

        visited_set = {(0,0)}

        while not((i==h) and (j==w)):
        
            # Specials cases when the agent is at the borders of the grid
            if j == w: # if the agent is at the right border, only move down
                self.agent.timer += self.grid[i+1,j] # update timer
                i+=1 # move down
                visited_set.add((i,j)) # mark the node as visited
                continue
            
            if i == h: # if the agent is at the lower border, only move right
                self.agent.timer += self.grid[i,j+1] # update timer
                j+=1 # move right
                visited_set.add((i,j)) # mark the node as visited
                continue
            
            # Compute the expected time for the right cell and its neighbours 
            # The neighbours are the cells to the right and above with the respect to the right cell
            time_right = self.compute_time(i+1, j+1) + self.compute_time(i, j+1) + self.compute_time(i, j+2)

            # Compute the expected time for the down cell and its neighbours 
            # The neighbours are the cells to the right and above with the respect to the down cell
            time_down = self.compute_time(i+1, j) + self.compute_time(i+2, j)+ self.compute_time(i+1, j+1)

            # Choose whether is it more convenient to move to right or below
            # I.e., move to the right when the right cell and its neighbours have a smaller
            # time compared to the down cell and its neighbours
            if time_right < time_down:
                self.agent.timer += self.grid[i,j+1] # update timer
                j+=1 # move right
            else:
                self.agent.timer += self.grid[i+1,j] # update timer
                i+=1 # move down

            visited_set.add((i,j)) # mark the node as visited
             
        self.visited_set = visited_set  

        self.agent.execution_time = ( time.time() - start  ) * 1000 # compute execution time in ms

    def compute_time(self, i,j):
        """
        This function is used to find the grid value of a specif cell, given the coordinates (i, j)
        It also checks whether the cell is inside the grid. If not, it returns 0
        """
        if (j > self.w - 1  or i > self.h -1 ): # checks if the cell is outside of the grid
            return 0 
        return self.grid[i,j]     