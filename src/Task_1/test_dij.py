import numpy as np 

def get_neighbours(current_node, visited_set):
    """
    This function finds the neighbour cells given the current cell. 
    Only unvisisted neighbours are considered

    """

    neighbours = list() # initialise the output
    i = current_node[0] # set vertical coordinate
    j = current_node[1] # set horizontal coordinate

                # QUA DEVI METTERE SELF!!!
    if i + 1 in range(4) and (i+1,j) not in visited_set:
        neighbours.append((i+1,j))
    if i - 1 in range(4) and (i-1,j) not in visited_set:
        neighbours.append((i-1,j))
    if j + 1 in range(4) and (i,j+1) not in visited_set:
        neighbours.append((i,j+1))
    if j - 1 in range(4) and (i,j-1) not in visited_set:
        neighbours.append((i,j-1))    
    return neighbours


a = np.array([[1,7,8,0],[4,8,3,9],[1,5,3,8],[0,3,5,6]])
target = (3,3)



# Initialise variables 
unvisited_list = [ (i,j) for i in range(4) for j in range(4)] # list containing all the nodes to be visited
distance_matrix = float("inf") * np.ones_like(a)  # type: ignore

current_node = (0,0) # initial location set as current node
distance_matrix[current_node] = 0 # set distance 0 for node (0,0)
visited_set = {current_node} # mark (0,0) as visited
unvisited_list.remove(current_node) # remove (0,0) from the unvisited nodes
visited_list = list()
#visited_list.append(((current_node),(current_node)))
timer = a[current_node]
counter = 0

while target not in visited_set: # loop until the target is visited

    neighbours = get_neighbours(current_node,visited_set) # get neighbour nodes 
    
    while len(neighbours)!=0: # loop for every neighbour
        temp_node = neighbours.pop()
        if distance_matrix[temp_node] > a[temp_node] + distance_matrix[current_node]: # update distance matrix with minimum values from neighbours
            distance_matrix[temp_node] = a[temp_node] + distance_matrix[current_node]
    
    distance_matrix_unvisited = [distance_matrix[node] for node in unvisited_list] # array with the values of all 
    next_node = unvisited_list[distance_matrix_unvisited.index(np.min(distance_matrix_unvisited))] # minimum distance for all unvisited nodes
    
    visited_set.add(next_node)  # type: ignore # mark next node as visited
    unvisited_list.remove(next_node) # remove next node to the unvisited list
    #visited_list.append(((current_node),(next_node)))   # type: ignore
    current_node = next_node # set the next node as the current

    counter +=1
    if counter == 1000:
        break
    
timer += distance_matrix[target]    
print(distance_matrix)    

print(visited_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    