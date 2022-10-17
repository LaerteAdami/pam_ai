# some_file.py
#import sys
#
## caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, "/Users/laerte/pam_ai/pam_ai/src/lab")
#
#import exercise as ex

from src.lab import exercise as ex

#import sys

BR = ex.broken_calculator(3)

a = 1
b = 2

correct_sum = ex.simple_math(a, b)

wrong_sum = BR.wrong_math(a, b)

print("Correct: {}, wrong: {}".format(correct_sum, wrong_sum))
