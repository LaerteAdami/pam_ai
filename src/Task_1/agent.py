class GameAgent():
    
    def __init__(self,i,j):
        self.i = i # horizontal position - from 0 to width
        self.j = j # vertical position - from 0 to height
        self.timer = 0 # total amout of time spent in the game

    def reset(self):
        self.i = 0
        self.j = 0
        self.timer = 0