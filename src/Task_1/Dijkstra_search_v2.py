class DijkstraSearch():

    def __init__(self, agent, random_grid):
        self.agent = agent
        self.grid = random_grid.grid
        self.w = random_grid.grid.shape[1] - 1
        self.h = random_grid.grid.shape[0] - 1
        self.count_steps = 0

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

        target = (self.h, self.w) # target cell, i.e., the bottom right one 

        # Initialise variables 
        unvisited_list = [ (i,j) for i in range(self.h+1) for j in range(self.w+1)] # list containing all the nodes to be visited
        distance_matrix = float("inf") * np.ones_like(self.grid)  # type: ignore

        current_node = (0,0) # initial location set as current node
        distance_matrix[current_node] = 0 # set distance 0 for node (0,0)
        visited_set = {current_node} # mark (0,0) as visited
        unvisited_list.remove(current_node) # remove (0,0) from the unvisited nodes

        self.agent.timer += self.grid[current_node]
        counter = 0

        while target not in visited_set: # loop until the target is visited
        
            neighbours = self.get_neighbours(current_node,visited_set) # get neighbour nodes 

            while len(neighbours)!=0: # loop for every neighbour
                temp_node = neighbours.pop()
                if distance_matrix[temp_node] > self.grid[temp_node] + distance_matrix[current_node]: # update distance matrix with minimum values from neighbours
                    distance_matrix[temp_node] = self.grid[temp_node] + distance_matrix[current_node]

            distance_matrix_unvisited = [distance_matrix[node] for node in unvisited_list] # array with the values of all 
            next_node = unvisited_list[distance_matrix_unvisited.index(np.min(distance_matrix_unvisited))] # minimum distance for all unvisited nodes

            visited_set.add(next_node)  # type: ignore # mark next node as visited
            unvisited_list.remove(next_node) # remove next node to the unvisited list
            #visited_list.append(((current_node),(next_node)))   # type: ignore
            current_node = next_node # set the next node as the current

            counter +=1
            #if counter == 1000:
            #    break
            
        self.agent.timer += distance_matrix[target]    

        self.visited_set = visited_set 
        self.agent.execution_time = (time.time() - start ) * 1000 # compute execution time in ms
        self.count_steps = counter

    def get_neighbours(self, current_node, visited_set):
        """
        This function finds the neighbour cells given the current cell. 
        Only unvisisted neighbours are considered

        """

        neighbours = list() # initialise the output
        i = current_node[0] # set vertical coordinate
        j = current_node[1] # set horizontal coordinate

        if i + 1 in range(self.h+1) and (i+1,j) not in visited_set:
            neighbours.append((i+1,j))
        if i - 1 in range(self.h+1) and (i-1,j) not in visited_set:
            neighbours.append((i-1,j))
        if j + 1 in range(self.w+1) and (i,j+1) not in visited_set:
            neighbours.append((i,j+1))
        if j - 1 in range(self.w+1) and (i,j-1) not in visited_set:
            neighbours.append((i,j-1))    
        return neighbours    