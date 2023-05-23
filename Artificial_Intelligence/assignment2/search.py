#search
# זהו הפתרון של תרגיל 1 והוא מימוש בפייתון לפסיאודוקוד שהוצג בכיתה

import state
import frontier

def search(n):
    s=state.create(n)
    print(s)
    f=frontier.create(s)
    while not frontier.is_empty(f):
        s=frontier.remove(f)
        print(s)
        if state.is_target(s):
            print("total states: ", frontier.totalStates(f))
            print("max queue len: ", frontier.maxQueueLenght(f))
            return s
            
        ns=state.get_next(s)
        for i in ns:
            frontier.insert(f,i)

    print("total states: ", frontier.totalStates(f))
    print("max queue len: ", frontier.maxQueueLenght(f))
    return 0
    
    
def main():
    search(4)

if __name__ == "__main__":
    main()   