import numpy as np
import time


class DijkstraSearch:
    def __init__(self, agent, random_grid):
        self.agent = agent
        self.grid = random_grid.grid
        self.w = random_grid.grid.shape[1] - 1
        self.h = random_grid.grid.shape[0] - 1

    def compute_path(self, get_path=False):
        """
        This method should decide in which direction to move by evaluating the cell to the right
        and below the selected cell (excluding special cases at the edges)
        For this cell, we evaluate the sum of the cell + the subsequent cells below and to the right
        The direction chosen is the one with the smallest sum
        In this way we can evaluate with a little more depth the best direction for each step
        """

        start = time.time()  # variable used to store the initial time.
        # It will be use to compute the total execution time of the search

        target = (self.h, self.w)  # target cell, i.e., the bottom right one

        # Initialise variables
        unvisited_list = [
            (i, j) for i in range(self.h + 1) for j in range(self.w + 1)
        ]  # list containing all the nodes to be visited
        distance_matrix = float("inf") * np.ones_like(self.grid)  # type: ignore

        current_node = (0, 0)  # initial location set as current node
        distance_matrix[current_node] = 0  # set distance 0 for node (0,0)
        visited_set = {current_node}  # mark (0,0) as visited
        unvisited_list.remove(current_node)  # remove (0,0) from the unvisited nodes

        self.agent.timer += self.grid[current_node]

        while target not in visited_set:  # loop until the target is visited

            neighbours = self.get_neighbours(
                current_node, visited_set
            )  # get neighbour nodes

            while len(neighbours) != 0:  # loop for every neighbour
                temp_node = neighbours.pop()
                if (
                    distance_matrix[temp_node]
                    > self.grid[temp_node] + distance_matrix[current_node]
                ):  # update distance matrix with minimum values from neighbours
                    distance_matrix[temp_node] = (
                        self.grid[temp_node] + distance_matrix[current_node]
                    )

            distance_matrix_unvisited = [
                distance_matrix[node] for node in unvisited_list
            ]  # array with the values of all
            next_node = unvisited_list[
                distance_matrix_unvisited.index(np.min(distance_matrix_unvisited))
            ]  # minimum distance for all unvisited nodes

            visited_set.add(next_node)  # type: ignore # mark next node as visited
            unvisited_list.remove(next_node)  # remove next node to the unvisited list
            current_node = next_node  # set the next node as the current

        self.agent.timer += int(distance_matrix[target])

        if get_path:
            self.visited_set = self.get_path(unvisited_list, distance_matrix)

        self.agent.execution_time = (
            time.time() - start
        ) * 1000  # compute execution time in ms

    def get_neighbours(self, current_node, lookup_set):
        """
        This function finds the neighbour cells given the current cell.
        Only neighbours not in the lookup_set are considered

        """

        neighbours = list()  # initialise the output
        i = current_node[0]  # set vertical coordinate
        j = current_node[1]  # set horizontal coordinate

        if (
            i + 1 in range(self.h + 1) and (i + 1, j) not in lookup_set
        ):  # if adjacent node is in correct range and is not in lookup_set
            neighbours.append((i + 1, j))
        if (
            i - 1 in range(self.h + 1) and (i - 1, j) not in lookup_set
        ):  # if adjacent node is in correct range and is not in lookup_set
            neighbours.append((i - 1, j))
        if (
            j + 1 in range(self.w + 1) and (i, j + 1) not in lookup_set
        ):  # if adjacent node is in correct range and is not in lookup_set
            neighbours.append((i, j + 1))
        if (
            j - 1 in range(self.w + 1) and (i, j - 1) not in lookup_set
        ):  # if adjacent node is in correct range and is not in lookup_set
            neighbours.append((i, j - 1))
        return neighbours

    def get_path(self, unvisited_list, distance_matrix):
        """
        This function computes the best path found in compute_path
        Return the visited_set of cells in the best path

        """

        current_node = (self.h, self.w)  # start from the last node

        unvisited_set = set(unvisited_list)
        visited_set = set()
        visited_set.add(current_node)  # type: ignore

        while current_node != (0, 0):  # go on until the origin

            neighbours = self.get_neighbours(
                current_node, unvisited_set
            )  # find neighbours of the current cell which was not unvisited (i.e., visited)
            unvisited_set.add(current_node)  # type: ignore

            if len(neighbours) == 1:
                current_node = neighbours[
                    0
                ]  # if there is only one neighbour, it must be in the path. Set it as current node
            else:
                neighbours_distance = np.zeros(
                    len(neighbours)
                )  # if there is more than one neighbour, choose the one with minimum distance
                neighbours_distance = [
                    distance_matrix[neighbours[i]] for i in range(len(neighbours))
                ]
                current_node = neighbours[
                    neighbours_distance.index(np.min(neighbours_distance))
                ]

            visited_set.add(current_node)  # type: ignore # add the current_node to the visited_set

        return visited_set
