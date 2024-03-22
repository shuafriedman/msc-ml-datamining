'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

(C) Copyright 5783
Udi Adler       
Chaim Schendowich 

UI static class. This class has all the static methods necessary for executing the
various game options.
'''


import random
from game import *
from agent import *

class ui:
    # Prints the game menu.
    def print_menu():
        print("------------------------------------------------")
        print("Menu")
        print("------------------------------------------------")
        print("    1. Change preferences")
        print("    2. Train agents")
        print("    3. Play demo")
        print("    4. Exit")
        print("------------------------------------------------")

    # Shows the current values of the preferences and allows the user to
    # change them at will.
    # Preferences that are numbers need a numeric input.
    # Preferences that are ratios need an input string of the format "p:q".
    def update_prefs(prefs):
        done = False
        keys = ['','R[H2H]','R[Win]','Discount',\
                ('P[(1,3)(r,l)|r]','P[(1,3)(r,l)|l]'),\
                ('P[(4,6)(r,l)|r]','P[(4,6)(r,l)|l]'),\
                ('P[(4,2)(r,u)|r]','P[(4,2)(r,u)|u]'),\
                ('P[(2,6)(u,l)|u]','P[(2,6)(u,l)|l]'),\
                ('P[1|u]','P[1|!u]'),\
                ('P[3|u]','P[3|!u]')\
                ]
        while not done:
            # Print old values
            print("Current preferences:")
            print("1. Reward head to head:", \
                  prefs['R[H2H]'])
            print("2. Reward win:", \
                  prefs['R[Win]'])
            print("3. Discount factor:", \
                  prefs['Discount'])
            print("4. Ratio head to head on 2 (M:B):", \
                  str(prefs['P[(1,3)(r,l)|r]']) + ":" + \
                  str(prefs['P[(1,3)(r,l)|l]']))
            print("5. Ratio head to head on 5 (M:B):", \
                  str(prefs['P[(4,6)(r,l)|r]']) + ":" + \
                  str(prefs['P[(4,6)(r,l)|l]']))
            print("6. Fence ratio from 4 (M:B):", \
                  str(prefs['P[(4,2)(r,u)|r]']) + ":" + \
                  str(prefs['P[(4,2)(r,u)|u]']))
            print("7. Fence ratio from 6 (M:B):", \
                  str(prefs['P[(2,6)(u,l)|u]']) + ":" + \
                  str(prefs['P[(2,6)(u,l)|l]']))
            print("8. Barrier ratio from 1 (Y:N):", \
                  str(prefs['P[1|u]']) + ":" + \
                  str(prefs['P[1|!u]']))
            print("9. Barrier ratio from 3 (Y:N):", \
                  str(prefs['P[3|u]']) + ":" + \
                  str(prefs['P[3|!u]']))
            # Prompt user
            choice = input("Which preference do you want to change (0-exit)? ")
            option = int(choice)
            if option > 0 and option <= 9:
                value = input("What is the new value for the preference? ")
            if option == 0:                                                 # Exit
                done = True
            elif option == 1 or option == 2 or option == 3:                 # Numeric
                prefs[keys[option]] = float(value)
            elif option > 3 and option <= 9:                                # Ratio
                pair = value.split(':')
                prefs[keys[option][0]] = int(pair[0])
                prefs[keys[option][1]] = int(pair[1])
            else:                                                           # Bad input
                print("Invalid choice!")
        return prefs

    # Initializes the game and trains the agents
    def train_agents(prefs):
        # Initialize game
        the_game = game(prefs)

        # Train agents
        agent1 = agent(the_game,1)
        agent1.train()
        agent2 = agent(the_game,2)
        agent2.train()
        
        # Print training results
        agent1.print_optimal_strategies()
        agent2.print_optimal_strategies()

        # Return game and trained agents
        return (the_game,agent1,agent2)

    # Demonstrates a simulation of the game.
    # The simulation is a number of iterations of the game played according to
    # one of two possible strategies for each player:
    #     1. Optimal stragegy: the strategy learned in the training process.
    #     2. Random strategy: a uniform random possible action is chosen each turn.
    # After each iteration the rewards for Mario and Bowser are displayed.
    # After the last iteration the total numbers of wins, the average number of turns
    # and the total rewards are displayed.
    def play_demo(g,agent1,agent2):
        # Input demo definitions
        num_games = input("How many games do you want to simulate? ")
        try:
            num_games = int(num_games)
        except:
            print("Number of games must be an integer!")
            ui.play_demo(g,agent1,agent2)
            return

        p1_optimal = input("Do you want " + g.players[0].name + " to use optimal strategies (Y/N)? ")
        p2_optimal = input("Do you want " + g.players[1].name + " to use optimal strategies (Y/N)? ")

        p1_optimal = p1_optimal == 'Y' or p1_optimal == 'y'
        p2_optimal = p2_optimal == 'Y' or p2_optimal == 'y'        

        # Initialize counters and summers
        p1_wins = 0
        p2_wins = 0
        p1_total_reward = 0
        p2_total_reward = 0
        total_moves = 0

        # Demo loop
        for num_game in range(num_games):
            print(str(num_game+1) + '.', end = '')
            q1 = False  # Player 1 won
            q2 = False  # Player 2 won
            moves = 0
            p1_game_reward = 0
            p2_game_reward = 0
            game_state = (1,3)  # Mario always starts at state 1, Bowser at state 3
            print(game_state, end = '')
            # Game loop
            while not q1 and not q2:
                rand_result1 = random.randint(0,99)
                rand_result2 = random.randint(0,99)

                # Compute strategies and resulting state on success
                if p1_optimal:
                    s1 = agent1.optimal_strategies[(game_state[0],game_state[1])]
                else:
                    s1 = agent1.get_random_strategy(game_state[0])
                d1 = g.states[game_state[0]-1].outcomes[s1]

                if p2_optimal:                  
                    s2 = agent2.optimal_strategies[(game_state[1],game_state[0])]
                else:
                    s2 = agent2.get_random_strategy(game_state[1])
                d2 = g.states[game_state[1]-1].outcomes[s2]

                # Compute success proability
                p1_success = g.P(game_state[0],d1,s1,game_state[1],s2)
                p2_success = g.P(game_state[1],d2,s2,game_state[0],s1)

                # Compute outcome and rewards
                new_state = (game_state[0],game_state[1])
                r1 = g.R(game_state[0],game_state[0],s1,game_state[1],s2)
                if rand_result1 < p1_success*100:
                    r1 = g.R(game_state[0],d1,s1,game_state[1],s2)
                    new_state = (d1,new_state[1])
                r2 = g.R(game_state[1],game_state[1],s2,game_state[0],s1)
                if g.IsHeadToHead(game_state[0],s1,game_state[1],s2):
                    if rand_result1 >= 100 - p2_success*100:
                        r2 = g.R(game_state[1],d2,s2,game_state[0],s1)
                        new_state = (new_state[0],d2)
                else:
                    if rand_result2 < p2_success*100:
                        r2 = g.R(game_state[1],d2,s2,game_state[0],s1)
                        new_state = (new_state[0],d2)
                game_state = (new_state[0],new_state[1])

                p1_game_reward = p1_game_reward + r1
                p2_game_reward = p2_game_reward + r2

                # Check if there is a win
                if game_state[0] == 5:
                    q1 = True
                if game_state[1] == 5:
                    q2 = True

                # Output and advance move
                print(' --> ' + str(game_state), end = '')
                
                moves = moves + 1
            # (End game loop)

            # Save game stats
            if q1:
                p1_wins = p1_wins+1
            if q2:
                p2_wins = p2_wins+1

            p1_total_reward = p1_total_reward + p1_game_reward
            p2_total_reward = p2_total_reward + p2_game_reward
            total_moves = total_moves + moves
            print(" Reward = [" + str(p1_game_reward) + "," + str(p2_game_reward) + "]")
        # (End demo loop)

        # Print outputs
        average_moves_per_game = total_moves / num_games

        print(g.players[0].name + " wins: " + str(p1_wins))
        print(g.players[1].name + " wins: " + str(p2_wins))
        print("Average moves per game: " + str(average_moves_per_game))
        print("Total reward for " + g.players[0].name + ": " + str(p1_total_reward))
        print("Total reward for " + g.players[1].name + ": " + str(p2_total_reward))
