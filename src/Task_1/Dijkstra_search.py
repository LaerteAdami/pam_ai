class DijkstraSearch():

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
        import numpy as np
        import time

        start = time.time() # variable used to store the initial time. 
                            # It will be use to compute the total execution time of the search

       # Initiate variables
        unvisited_set = set([ (i,j) for i in range(self.h) for j in range(self.w)])
        distance_matrix = float("inf") * np.ones_like(self.grid)  # type: ignore
        
        # Set current node to (0,0)
        current_node = (0,0)
        distance_matrix[current_node] = 0
        self.agent.timer = self.grid[0,0] # initialise the timer of the agent
        
        # Initialise sets
        visited_set = {current_node}
        unvisited_set.remove(current_node)

        stop_condition = False

        while not stop_condition:

            i = current_node[0]
            j = current_node[1]
            neighborhood = list()
            
            if (i+1,j) in unvisited_set: self.update_distance(distance_matrix,i+1, j, current_node); neighborhood.append((i+1,j))
            if (i,j+1) in unvisited_set: self.update_distance(distance_matrix,i, j+1, current_node); neighborhood.append((i,j+1))
            #if j != (self.w-1) and (i-1,j) in unvisited_set: self.update_distance(distance_matrix, i-1, j, current_node); neighborhood.append((i-1,j))
            #if i != (self.h-1) and (i,j-1) in unvisited_set: self.update_distance(distance_matrix, i, j-1, current_node); neighborhood.append((i,j-1))

            find_min = list()
            for k in neighborhood:
                find_min.append(distance_matrix[k])

            next_node = neighborhood[find_min.index(min(find_min))]    

            unvisited_set.remove(next_node)
            visited_set.add(next_node)   # type: ignore 
            current_node = next_node
            stop_condition = current_node == (self.h-1,self.w-1)

            self.agent.timer += self.grid[next_node]

        self.visited_set = visited_set 
        self.execution_time = (time.time() - start ) * 1000 # compute execution time in ms

    def update_distance(self, distance_matrix, i, j, current_node):
        """
        update_distance is used to updated the distance_matrix with new distance values, smaller than
        the previous ones

        """
        if (j > self.w   or i > self.h or i < 0 or j < 0 ):
            return  
        distance_matrix[i,j] = self.grid[i,j] + distance_matrix[current_node]   
        return distance_matrix    