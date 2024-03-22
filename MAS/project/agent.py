'''
Mario vs. Bowser
Final project in the course: Multiagent Systems and Distributed Systems

 

Agent class. Represents an agent in a game. Can be trained for optimal strategies using
the Shapely value iteration algorithm. 
'''
 

import random
from game import *
    
class agent:
    # Initializer
    def __init__(self, game, player_number):
        self.game = game
        self.player = self.game.players[player_number-1] 
        self.optimal_strategies = {}

    def train(self):
        for state in self.game.states:
            for opponent_state in self.game.states:
                state_tuple = (state.name, opponent_state.name)
                best_action_value = float('-inf')
                best_action = None
                for action in state.actions:
                    action_value = self.evaluate_action(state, opponent_state, action)
                    if action_value > best_action_value:
                        best_action_value = action_value
                        best_action = action
                if best_action is not None:
                    self.optimal_strategies[state_tuple] = best_action

    def evaluate_action(self, state, opponent_state, action):
        discount_factor = self.game.prefs['Discount']
        expected_utility = 0
        for opp_action in opponent_state.actions:
            for next_state in self.game.states:
                transition_probability = self.game.P(state.name, next_state.name, action, opponent_state.name, opp_action)
                immediate_reward = self.game.R(state.name, next_state.name, action, opponent_state.name, opp_action)
                future_value = self.estimate_future_value(next_state, opponent_state)
                expected_utility += transition_probability * (immediate_reward + discount_factor * future_value)
        return expected_utility
    
    def estimate_future_value(self, state, opponent_state):
        # Simplified future value estimation
        future_value = 0
        for action in state.actions:
            for opp_action in opponent_state.actions:
                for next_state in self.game.states:
                    prob = self.game.P(state.name, next_state.name, action, opponent_state.name, opp_action)
                    reward = self.game.R(state.name, next_state.name, action, opponent_state.name, opp_action)
                    future_value += prob * reward
        return future_value

        return future_value / num_actions if num_actions > 0 else 0
    def get_random_strategy(self, state):
        state = self.game.states[state - 1]  # Adjusting index for 0-based indexing
        if not state.actions:
            return None  # or some placeholder indicating no action is possible
        return random.choice(state.actions)
    
    def print_optimal_strategies(self):
        if self.optimal_strategies:
            print(f"Optimal strategies for {self.player.name}:")
            for state, action in self.optimal_strategies.items():
                print(f"State {state}: {action}")
        else:
            print(f"No strategies computed for {self.player.name}.")
