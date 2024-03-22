'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

(C) Copyright 5783
Udi Adler
Chaim Schedowich

Player class. This class stores information relevant for a player in the game,
namely, Mario or Bowser. The information stored is the name of the player and a list of
states in which the player can be.
'''


class player:
    # Initializer
    def __init__(self,name='player',states=[]):
        self.name = name
        self.states = states
        if not type(states) is list:
            self.states = []

    # A method for adding a legal state to the player's state list.
    def add_state(self,state):
        if state in self.states:
            return
        self.states.append(state)
