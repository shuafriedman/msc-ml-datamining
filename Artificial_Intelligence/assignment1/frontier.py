'''
# יוצר מחסנית ״משוכללת״
[stack, max. depth, init. state, try next level?]
stack - a simple stack as defined at stack.py
max.depth - the current search depth of ID
init.state - the initial state of the problem
try next level - is there a reason to search deeper
'''
import stack
import state

def create(x):
    s=stack.create(x)
    return [ s,1,x,False,0]

def is_empty(s):
    return stack.is_empty(s[0]) and not s[3] # stack is empty and try next level is false

def insert(s,x, total):
    if state.path_len(x)<=s[1]: # check if x is not too deep
        stack.insert(s[0],x)    # insert x to stack
        s[-1] = total
    else:
        s[3]=True               # there is a reason to search deeper if needed
    
def remove(s):
    if stack.is_empty(s[0]):    # check is there are no states in the stack
        if s[3]:                # check if there is a reason to search deeper
            s[1]+=1             # increase search depth
            s[3]=False          # meanwhile there is no evidence to  a need to search deeper
            return s[2]         # return the initial state
        else:
            return 0
    return stack.remove(s[0])   # if there are items in the stack ...

    
