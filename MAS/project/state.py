'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

(C) Copyright 5783
Udi Adler 
Chaim Schendowich 

State class. This class stores information about a state in the game.
Each state has a number of valid actions. The actions are stored in the
list actions and the resulting state per action are stored in the dict outcomes.
'''


class state:
    # Initializer
    def __init__(self,name=0,actions=None,outcomes=None):
        self.name = name
        self.actions = actions
        if not type(actions) is list:
            self.actions = []
        self.outcomes = outcomes
        if not type(outcomes) is dict:
            self.outcomes = {}

    # A setter method the outcome of a given action.
    def add_outcome(self,action,outcome):
        if not action in self.actions:
            self.actions.append(action)

        self.outcomes[action] = outcome
