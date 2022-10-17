import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/Users/laerte/pam_ai/pam_ai/src')


from Task_1.agent import game_agent

ag = game_agent(0,0)
ag.add_step(10)
print(ag.timer)
