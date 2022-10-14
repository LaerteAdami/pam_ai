import exercise as ex

import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/PAM_AI/src/lab')

BR = ex.broken_calculator(3)

a = 1
b = 2

correct_sum = ex.simple_math(a,b)

wrong_sum = BR.wrong_math(a,b)

print("Correct: {}, wrong: {}".format(correct_sum,wrong_sum))
