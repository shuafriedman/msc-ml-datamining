'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

(C) Copyright 5783
Udi Adler 
Chaim Schendowich 

Game class. Contains all necessary information about the game structure and rules.
More specifically:
    1. Definition of the game states (1-6).
    2. Definition of the players (1-2).
    3. Definition of the transition probabilities (P).
    4. Definition of the transition rewards (R).
'''


from state import *
from player import *

class game:
    # Initializer
    # A preferences list is a required argument.
    def __init__(self,prefs):
        self.prefs = prefs
        self.initialize_board()

        # Initialize probabilities for single player stochatic transitions.
        self.p1 = {}
        self.p1[(1,"up")] = {}
        self.p1[(1,"up")][4] = self.prefs['P[1|u]']/(self.prefs['P[1|u]']+self.prefs['P[1|!u]'])
        self.p1[(1,"up")][1] = self.prefs['P[1|!u]']/(self.prefs['P[1|u]']+self.prefs['P[1|!u]'])
        self.p1[(3,"up")] = {}
        self.p1[(3,"up")][6] = self.prefs['P[3|u]']/(self.prefs['P[3|u]']+self.prefs['P[3|!u]'])
        self.p1[(3,"up")][3] = self.prefs['P[3|!u]']/(self.prefs['P[3|u]']+self.prefs['P[3|!u]'])

        # Initialize probabilities for two player stochastic transitions.
        self.p2 = {}
        self.p2[(1,3)] = {}
        self.p2[(1,3)][("right","left")] = {}
        self.p2[(1,3)][("right","left")][1] = self.prefs['P[(1,3)(r,l)|l]']/(self.prefs['P[(1,3)(r,l)|r]']+self.prefs['P[(1,3)(r,l)|l]'])
        self.p2[(1,3)][("right","left")][2] = self.prefs['P[(1,3)(r,l)|r]']/(self.prefs['P[(1,3)(r,l)|r]']+self.prefs['P[(1,3)(r,l)|l]'])
        self.p2[(3,1)] = {}
        self.p2[(3,1)][("left","right")] = {}
        self.p2[(3,1)][("left","right")][2] = self.prefs['P[(1,3)(r,l)|l]']/(self.prefs['P[(1,3)(r,l)|r]']+self.prefs['P[(1,3)(r,l)|l]'])
        self.p2[(3,1)][("left","right")][3] = self.prefs['P[(1,3)(r,l)|r]']/(self.prefs['P[(1,3)(r,l)|r]']+self.prefs['P[(1,3)(r,l)|l]'])
        self.p2[(4,6)] = {}
        self.p2[(4,6)][("right","left")] = {}
        self.p2[(4,6)][("right","left")][4] = self.prefs['P[(4,6)(r,l)|l]']/(self.prefs['P[(4,6)(r,l)|r]']+self.prefs['P[(4,6)(r,l)|l]'])
        self.p2[(4,6)][("right","left")][5] = self.prefs['P[(4,6)(r,l)|r]']/(self.prefs['P[(4,6)(r,l)|r]']+self.prefs['P[(4,6)(r,l)|l]'])
        self.p2[(6,4)] = {}
        self.p2[(6,4)][("left","right")] = {}
        self.p2[(6,4)][("left","right")][5] = self.prefs['P[(4,6)(r,l)|l]']/(self.prefs['P[(4,6)(r,l)|r]']+self.prefs['P[(4,6)(r,l)|l]'])
        self.p2[(6,4)][("left","right")][6] = self.prefs['P[(4,6)(r,l)|r]']/(self.prefs['P[(4,6)(r,l)|r]']+self.prefs['P[(4,6)(r,l)|l]'])
        self.p2[(4,2)] = {}
        self.p2[(4,2)][("right","up")] = {}
        self.p2[(4,2)][("right","up")][4] = self.prefs['P[(4,2)(r,u)|u]']/(self.prefs['P[(4,2)(r,u)|r]']+self.prefs['P[(4,2)(r,u)|u]'])
        self.p2[(4,2)][("right","up")][5] = self.prefs['P[(4,2)(r,u)|r]']/(self.prefs['P[(4,2)(r,u)|r]']+self.prefs['P[(4,2)(r,u)|u]'])
        self.p2[(2,4)] = {}
        self.p2[(2,4)][("up","right")] = {}
        self.p2[(2,4)][("up","right")][2] = self.prefs['P[(4,2)(r,u)|r]']/(self.prefs['P[(4,2)(r,u)|r]']+self.prefs['P[(4,2)(r,u)|u]'])
        self.p2[(2,4)][("up","right")][5] = self.prefs['P[(4,2)(r,u)|u]']/(self.prefs['P[(4,2)(r,u)|r]']+self.prefs['P[(4,2)(r,u)|u]'])
        self.p2[(2,6)] = {}
        self.p2[(2,6)][("up","left")] = {}
        self.p2[(2,6)][("up","left")][2] = self.prefs['P[(2,6)(u,l)|l]']/(self.prefs['P[(2,6)(u,l)|u]']+self.prefs['P[(2,6)(u,l)|l]'])
        self.p2[(2,6)][("up","left")][5] = self.prefs['P[(2,6)(u,l)|u]']/(self.prefs['P[(2,6)(u,l)|u]']+self.prefs['P[(2,6)(u,l)|l]'])
        self.p2[(6,2)] = {}
        self.p2[(6,2)][("left","up")] = {}
        self.p2[(6,2)][("left","up")][5] = self.prefs['P[(2,6)(u,l)|l]']/(self.prefs['P[(2,6)(u,l)|u]']+self.prefs['P[(2,6)(u,l)|l]'])
        self.p2[(6,2)][("left","up")][6] = self.prefs['P[(2,6)(u,l)|u]']/(self.prefs['P[(2,6)(u,l)|u]']+self.prefs['P[(2,6)(u,l)|l]'])

        # Initialize rewards.
        self.r = {}
        self.r[((1,3),("right","left"),2)] = self.prefs['R[H2H]']
        self.r[((3,1),("left","right"),2)] = self.prefs['R[H2H]']
        self.r[((1,3),("right","left"),1)] = -self.prefs['R[H2H]']
        self.r[((3,1),("left","right"),3)] = -self.prefs['R[H2H]']
        self.r[((4,6),("right","left"),5)] = self.prefs['R[H2H]']+self.prefs['R[Win]']
        self.r[((6,4),("left","right"),5)] = self.prefs['R[H2H]']+self.prefs['R[Win]']
        self.r[((4,6),("right","left"),4)] = -self.prefs['R[H2H]']-self.prefs['R[Win]']
        self.r[((6,4),("left","right"),6)] = -self.prefs['R[H2H]']-self.prefs['R[Win]']

    # Board initialization method.
    def initialize_board(self):
        # Initialize states.
        state1 = state(1,actions=["right","up"])
        state1.add_outcome("right",2)
        state1.add_outcome("up",4)
        state2 = state(2,actions=["up"])
        state2.add_outcome("up",5)
        state3 = state(3,actions=["left","up"])
        state3.add_outcome("left",2)
        state3.add_outcome("up",6)
        state4 = state(4,actions=["right"])
        state4.add_outcome("right",5)
        state5 = state(5,actions=[])
        state6 = state(6,actions=["left"])
        state6.add_outcome("left",5)
        self.states = [state1,state2,state3,state4,state5,state6]

        # Initialize players.
        player1 = player("Mario",[1,2,4,5])
        player1.start = 1
        player2 = player("Bowser",[2,3,5,6])
        player2.start = 3
        self.players = [player1,player2]

    # Boolean method that tests if the transitions of the players are contradictory,
    # resulting in a head-to-head situation.
    def IsHeadToHead(self,src,act,opp_s,opp_a):
        return self.states[src-1].outcomes[act] == self.states[opp_s-1].outcomes[opp_a]

    # Transition probability method.
    def P(self,src,dest,act,opp_s,opp_a):
        # Single player transition.
        if (src,act) in self.p1 and \
           dest in self.p1[(src,act)]:
            return self.p1[(src,act)][dest]

        # Two player transition.
        if (src,opp_s) in self.p2 and \
           (act,opp_a) in self.p2[(src,opp_s)] and \
           dest in self.p2[(src,opp_s)][(act,opp_a)]:
            return self.p2[(src,opp_s)][(act,opp_a)][dest]

        # Legal deterministic action.
        if act in self.states[src-1].actions and \
           self.states[src-1].outcomes[act] == dest:
            return 1

        return 0

    # Transition reward method.
    def R(self,src,dest,act,opp_s,opp_a):
        # Head to head transition.
        if ((src,opp_s),(act,opp_a),dest) in self.r:
            return self.r[((src,opp_s),(act,opp_a),dest)]

        # Player wins transition,
        if act in self.states[src-1].actions and \
           self.states[src-1].outcomes[act] == dest and \
           dest == 5:
            return self.prefs['R[Win]']

        # Opponent wins transition.
        if opp_a in self.states[opp_s-1].actions and \
           self.states[opp_s-1].outcomes[opp_a] == 5:
            return -self.prefs['R[Win]']

        # Anything else.
        return 0
