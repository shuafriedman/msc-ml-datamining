import frontier
import state
import numpy as np
n= 3

f = frontier.create(state.create(n))
total=0
while not frontier.is_empty(f):
    print('Checking Frontier')
    print(f)
    s = frontier.remove(f)
    print('Removed from Frontier:')
    print(np.array(s[0]).reshape(n,n))
    if state.is_target(s):
        print("Solution found:", s )
        break
    ns = state.get_next(s)
    print('Adding to frontier')
    for i in range(len(ns)):
        print('Add')
        print(np.array(ns[i][0]).reshape(n,n))
        total +=1
        frontier.insert(f,ns[i], total)
        # print(f)
# print("No solution found")
