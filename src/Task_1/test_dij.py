import numpy as np 

def get_neighbours(current_node):

    neighbours = list()
    i = current_node[0]
    j = current_node[1]

    if i + 1 in range(4):
        neighbours.append((i+1,j))
    if i - 1 in range(4):
        neighbours.append((i-1,j))
    if j + 1 in range(4):
        neighbours.append((i,j+1))
    if j - 1 in range(4):
        neighbours.append((i,j-1))    
    return neighbours


a = np.array([[1,7,8,0],[4,8,3,9],[1,5,3,8],[0,3,5,6]])


target = (3,3)
current_node = (0,0)

#unvisited_set = set([ (i,j) for i in range(4) for j in range(4)])
distance_matrix = float("inf") * np.ones_like(a)  # type: ignore
distance_matrix[current_node] = 0
visited_set = {current_node}

neighbours = get_neighbours(current_node)

while len(neighbours)!=0:

    temp_node = neighbours.pop()
    if distance_matrix[temp_node] > a[temp_node] + distance_matrix[current_node]:
        distance_matrix[temp_node] = a[temp_node] + distance_matrix[current_node]

#while current_node != target:
#    break
#    pass
#    
print(distance_matrix)














