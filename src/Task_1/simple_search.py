class simple_search():

    def __init__(self, agent, grid):
        self.agent = agent
        self.grid = grid
        self.w = grid.grid.shape[1]
        self.h = grid.grid.shape[0]

    def evaluate_direction(self, grid):
        """
        Questo metodo dovrebbe decidere in quale direzione muoversi
        Valuata la cella a destra e sotto di quella selezionata (esclusi casi particolati ai bordi)
        Per questa celle, valutiamo la somma della cella + le successive celle sotto e destra
        La direzione scelte è quella che ha la minore somma 
        In questo modo possiamo valuatare con un po' più di profondità la direzione migliore per ogni step
        """

        right_value = grid[self.agent.j][self.agent.i+1] #prendi il valore orizzontale
        down_value = grid[self.agent.j+1][self.agent.i] #prendi il valore sotto

        if right_value < down_value:
            return "r"
        else:
            return "d"     





