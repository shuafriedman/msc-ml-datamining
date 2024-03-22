'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

(C) Copyright 5783
Udi Adler      
Chaim Schendowich 

Main module. Run this module to activate the program.
'''


from ui import *

done = False

# Here we initialize the preferences for the game.
prefs = {
    'R[H2H]':0.5,
    'R[Win]':1,
    'Discount':0.9, #changes by Rina Azoulay.
    'P[(1,3)(r,l)|r]':2,
    'P[(1,3)(r,l)|l]':3,
    'P[1|u]':2,
    'P[1|!u]':1,
    'P[3|u]':1,
    'P[3|!u]':2,
    'P[(4,6)(r,l)|r]':2,
    'P[(4,6)(r,l)|l]':3,
    'P[(4,2)(r,u)|r]':3,
    'P[(4,2)(r,u)|u]':1,    
    'P[(2,6)(u,l)|u]':1,
    'P[(2,6)(u,l)|l]':3
    }

# The game and the agents are initialized to None to force the user to train
# the agents before playing the demo.
(g,a1,a2) = (None,None,None)

while not done:
    ui.print_menu()
    choice = input("Your choice: ")
    if choice == '1':                           # 1. Update preferences
        prefs = ui.update_prefs(prefs)
    elif choice == '2':                         # 2. Train agents
        (g,a1,a2) = ui.train_agents(prefs)
    elif choice == '3':                         # 3. Play demo
        if g != None:
            ui.play_demo(g,a1,a2)
        else:
            print("Cannot play demo without trained agents!")
    elif choice == '4':                         # 4. Exit
        done = True
    else:
        print("Invalid choice!")
