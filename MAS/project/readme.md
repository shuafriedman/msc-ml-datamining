The provided code implements a Q-learning agent for a two-player game environment, specifically designed for characters named Mario and Bowser. This agent is part of a reinforcement learning system that learns to make decisions—selecting actions to maximize future rewards—based on interactions with the game environment. Here's an overview of the overall code structure and functionality:

1. **Initialization (`__init__`)**: The agent is initialized with key parameters and structures for learning, including the game environment, the player it controls (Mario or Bowser), learning rate (`alpha`), discount factor (`gamma`), exploration rate (`epsilon`), and dictionaries for storing Q-values and optimal strategies.

2. **Training (`train`)**: The agent undergoes training over a specified number of episodes. Each episode is a complete run of the game from start to end (or until a terminal state is reached), during which the agent interacts with the environment to learn from the outcomes of its actions.

3. **Gameplay Simulation (`play_game`)**: Simulates a game episode, where the agent and its opponent take turns performing actions based on the current game state. The agent selects actions using either an exploration strategy (random choice) or an exploitation strategy (choosing the action with the highest Q-value), depending on the exploration rate (`epsilon`). The opponent's actions are chosen randomly for simplicity.

4. **Action Selection (`choose_action` and `choose_opponent_action`)**: These methods determine the actions to be taken by the agent and its opponent, respectively. The agent's action is chosen based on an ε-greedy policy, which balances between exploring new actions and exploiting the known best actions.

5. **Q-value Update (`update_q_values`)**: After each action, the agent updates its Q-values based on the received reward and the estimated future rewards, using the Temporal Difference (TD) learning formula. This update process refines the agent's strategy over time, aiming to maximize total rewards.

6. **Optimal Strategy Determination (`determine_optimal_strategies`)**: Post-training, the agent analyzes its learned Q-values to determine the optimal action for each encountered state, which are then stored in `optimal_strategies`.

7. **Random Strategy Utility (`get_random_strategy`)**: A helper function used for selecting a random action from the set of legal actions in the current state, primarily for exploration purposes.

8. **Optimal Strategies Output (`print_optimal_strategies`)**: This method prints the optimal action for each state as determined by the agent, useful for debugging and analysis of the learned strategies.

Overall, this code represents a Q-learning-based approach to developing an autonomous agent capable of learning optimal gameplay strategies through repeated interactions with a two-player game environment. The agent aims to learn from both the successes and failures of its actions, adjusting its strategy to improve performance over time.

*Changes made to other files:

The modifications made to the `game` class were crucial for integrating a Q-learning approach within the existing game framework. These changes were specifically designed to support the development of a reinforcement learning agent capable of learning optimal strategies through interactions with the game environment. Here’s a concise overview of the changes and their significance:

### Introduction of Game State Management:
- **`reset_game` Method**: A new method was introduced to reset the game to its initial state, ensuring that each training episode or game simulation starts from a consistent starting point. This method sets the foundation for episodic learning, which is a core concept in reinforcement learning.

- **`step` Method**: This method simulates the progression of the game by one step based on the actions taken by both players. It calculates the next state of the game and the immediate reward resulting from the actions, and determines whether the game has reached a terminal state. This method is essential for the agent to interact with the environment and receive feedback in the form of state transitions and rewards.

### Enhanced Action and State Handling:
- **`get_legal_actions` Method**: Adjusted to return legal actions available to both Mario and Bowser for any given game state. This method supports the agent's decision-making process by providing it with the possible actions it can take, which is necessary for both exploring the action space and exploiting known strategies.

- **`get_all_states` Method**: Introduced to provide a comprehensive list of all possible game states. This is particularly useful for initializing Q-values across the state-action space and for evaluating the agent's performance across different scenarios.

### Integration with Q-learning:
These additions and modifications were specifically designed to support the Q-learning algorithm implemented in the `agent.py` file. The Q-learning method requires a well-defined environment where it can observe states, take actions, receive rewards, and understand when an episode (game) has concluded. The changes made to the `game` class provide this structured environment, allowing for effective implementation of the following Q-learning components:

- **Episodic Learning**: By resetting the game at the start of each training episode and progressing through states using the `step` method, the agent can learn from a series of interactions (episodes) with the game environment.

- **Exploration and Exploitation**: The `get_legal_actions` method enables the agent to explore the action space by understanding the actions available at each state, which is crucial for balancing exploration with exploitation.

- **Reward Feedback**: The reward system, defined through the `step` method's reward calculation, provides immediate feedback to the agent about the consequences of its actions, enabling it to learn strategies that maximize cumulative rewards.

- **Learning from Transitions**: The `step` method's ability to compute next states and rewards for action pairs allows the agent to learn the dynamics of the game environment implicitly, which is essential for updating Q-values and refining strategies.

In summary, the modifications to the `game` class were specifically tailored to accommodate and support the Q-learning approach, facilitating the agent's ability to learn through interaction with a structured and responsive game environment.
