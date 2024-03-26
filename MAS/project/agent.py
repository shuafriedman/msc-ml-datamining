import random
class agent:
    def __init__(self, game, player_number):
        self.game = game
        self.player_number = player_number
        self.q_values = {}  # Q-values, keyed by (state, action) pairs.
        self.alpha = 0.1  # Learning rate.
        self.gamma = 0.9  # Discount factor.
        self.epsilon = 0.1  # Exploration rate.
        self.optimal_strategies = {}  # Optimal strategies, keyed by state.
    def train(self, episodes=1000):
        for episode in range(episodes):
            self.play_game()

    def play_game(self):
        current_state = self.game.reset_game()
        done = False
        while not done:
            # Assuming player 1 is Mario and player 2 is Bowser for simplicity
            if self.player_number == 1:
                mario_action = self.choose_action(current_state, self.player_number)
                bowser_action = self.choose_opponent_action(current_state, 2)  # Implement this
                action = (mario_action, bowser_action)
            else:
                mario_action = self.choose_opponent_action(current_state, 1)  # Implement this
                bowser_action = self.choose_action(current_state, self.player_number)
                action = (mario_action, bowser_action)
            
            next_state, reward, done = self.game.step(action)
            self.update_q_values(current_state, action, reward, next_state)
            current_state = next_state

    def choose_action(self, state, player_number):
        # Choose action for the agent's player
        legal_actions = self.game.get_legal_actions(state)[player_number - 1]
        if random.random() < self.epsilon:  # Explore
            return random.choice(legal_actions)
        else:  # Exploit
            q_values = [self.q_values.get((state, a), 0) for a in legal_actions]
            max_q_value = max(q_values)
            max_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q_value]
            return random.choice(max_actions)
        
    def choose_opponent_action(self, state, player_number):
        # Optionally implement logic to choose an action for the opponent
        # For simplicity, this could randomly choose an action
        legal_actions = self.game.get_legal_actions(state)[player_number - 1]
        return random.choice(legal_actions)
    
    def update_q_values(self, state, action, reward, next_state):
        # Assuming action is a tuple (mario_action, bowser_action)
        # and the agent's player_number determines which set of actions to consider.
        player_actions = self.game.get_legal_actions(next_state)[self.player_number - 1]
        
        # Now we need to adjust the Q-value update logic to work correctly with this structure.
        # If the agent represents Mario (player_number 1), then we use mario_action; if Bowser (player_number 2), then bowser_action.
        # For simplicity, let's just continue with the assumption that we're only considering
        # the agent's own actions for Q-value updates.
        best_q_value = float('-inf')
        for a in player_actions:
            # Construct a hypothetical action tuple for Q-value lookup, assuming the opponent does nothing or some default.
            # This may need adjustment based on how you've structured actions and states.
            hypothetical_action = (a, None) if self.player_number == 1 else (None, a)
            q_value = self.q_values.get((next_state, hypothetical_action), 0)
            if q_value > best_q_value:
                best_q_value = q_value

        # Update the Q-value for the state-action pair
        current_q_value = self.q_values.get((state, action), 0)
        td_target = reward + self.gamma * best_q_value
        td_delta = td_target - current_q_value
        self.q_values[(state, action)] = current_q_value + self.alpha * td_delta
        self.determine_optimal_strategies()

    def determine_optimal_strategies(self):
        for state in self.game.get_all_states():  # Assuming this method provides all possible states.
            legal_actions = self.game.get_legal_actions(state)[self.player_number - 1]
            best_action = None
            best_q_value = float('-inf')
            for action in legal_actions:
                q_value = self.q_values.get((state, action), 0)  # Retrieve Q-value for each action.
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            if best_action is not None:
                self.optimal_strategies[state] = best_action
                
    def get_random_strategy(self, state):
        # Retrieve the legal actions for the agent based on its player number and the current state
        legal_actions = self.game.get_legal_actions(state)[self.player_number - 1]

        # If there are no legal actions (e.g., in a terminal state), return None or a default action
        if not legal_actions:
            return None
        
        # Select and return a random action from the list of legal actions
        return random.choice(legal_actions)
    
    def print_optimal_strategies(self):
        for state in self.game.get_all_states():  # Assuming this method returns all possible states
            legal_actions = self.game.get_legal_actions(state)[self.player_number - 1]
            if not legal_actions:  # If there are no legal actions, skip to the next state
                continue
            best_action = max(legal_actions, key=lambda a: self.q_values.get((state, a), 0))
            print(f"State {state}: Best action: {best_action}")