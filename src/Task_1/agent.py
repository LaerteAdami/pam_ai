class GameAgent():
    
    def __init__(self,i,j):
        self.i = i # horizontal position - from 0 to width
        self.j = j # vertical position - from 0 to height
        self.timer = 0 # total amout of time spent in the game
        self.path = "" # string to keep track of all the movements made

    def add_step(self, step = int):

        self.timer +=  step
