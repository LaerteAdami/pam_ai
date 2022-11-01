class RandomSearch():

    random = __import__('random')
    def __init__(self, agent, random_grid, max_steps):
        self.agent = agent
        self.grid = random_grid.grid
        self.w = random_grid.grid.shape[1]
        self.h = random_grid.grid.shape[0]
        self.max_steps = max_steps

    def compute_path(self):
        """
        This method aims to reach the final grid cell through a random series of movements of the agent. 
        At each step, a random direction is chosen to travel between the adjacent cells.
        Since a random walk could take a very larger amount of steps, the computation is bounded 
        with a maximum value of steps, i.e., max_steps
        """
        import time

        start = time.time() # variable used to store the initial time. 
                            # It will be use to compute the total execution time of the search

        self.agent.timer = self.grid[0,0] # initialise the timer of the agent
        i = self.agent.i # horizontal position of the agent at the start
        j = self.agent.j # vertical position of the agent at the start
        
        current_node = (0,0) # initialise the starting position of the agent
        visited_set = {current_node}
        counter = 0 # counter to evaluate the number of steps of the path

        while not((current_node[0]==(self.h - 1)) and (current_node[1]==(self.w - 1))):
        
            next_node = self.random_step(current_node) # select a random step from the current node

            self.agent.timer += self.grid[next_node] # update timer

            current_node = next_node # set the next node as the current node

            visited_set.add(current_node)  # type: ignore

            counter += 1 # update counter

            if counter == self.max_steps: # when the maximum number of steps has been reached, the method stops
                break

        self.visited_set = visited_set 

        self.execution_time = (time.time() - start ) * 1000 # compute execution time in ms
            
    def random_step(self, current_node):
        """
        This function is employed to determine randomly the next step the agent will take 
        Input: current_node
        Output: next_node 

        """

        import random

        random_direction = random.randrange(4) 
        # select one random direction, encoded as follows:
        # random_direction = 0 --> move down
        # random_direction = 1 --> move right
        # random_direction = 2 --> move up
        # random_direction = 3 --> move left
                
        i = current_node[0]
        j = current_node[1] 
        next_node = current_node

        if random_direction == 0: # step down
            if i == self.h - 1: # when the agent is the the lower border, skip the next step
                pass 
            else: 
                next_node = (i+1,j)
        elif random_direction == 1: # step right 
            if j == self.w - 1: # when the agent is the the right border, skip the next step
                pass
            else:
                next_node = (i,j+1)   
        elif random_direction == 2: # step up 
            if i == 0: # when the agent is the the upper border, skip the next step
                pass
            else:
                next_node = (i-1,j) 
        else: # step left 
            if j == 0: # when the agent is the the left border, skip the next step
                pass
            else:
                next_node = (i,j-1)   
          
        return next_node
