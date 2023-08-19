import numpy as np
class SS_RTB_QLearning:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        # Initialize Q-table with zeros for each state-action pair
        self.q_table = np.zeros((state_space_size, action_space_size))

        # Learning rate for Q-value updates
        self.learning_rate = learning_rate

        # Discount factor for future rewards
        self.discount_factor = discount_factor

        # Exploration rate determines the probability of choosing a random action
        self.exploration_rate = exploration_rate

        # Factor by which exploration rate decays each time step
        self.exploration_decay = exploration_decay

        # Size of the state space
        self.state_space_size = state_space_size

        # Size of the action space
        self.action_space_size = action_space_size

    def choose_action(self, state):
        # Decide whether to explore or exploit based on the exploration rate
        if np.random.rand() < self.exploration_rate:
            # Explore: choose a random action from the action space
            return np.random.choice(self.action_space_size)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        # Determine the best action for the next state (used for Q-value update)
        best_next_action = np.argmax(self.q_table[next_state, :])

        # Current Q-value estimate for the taken action in the given state
        q_predict = self.q_table[state, action]

        # Updated Q-value target based on received reward and future rewards
        q_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]

        # Update the Q-value for the taken action in the given state
        self.q_table[state, action] += self.learning_rate * (q_target - q_predict)

        # Decay the exploration rate
        self.exploration_rate *= self.exploration_decay

# The comments added to each line or block explain the purpose and functionality of the code.

# Sample historical data (for demonstration purposes)
# Each tuple is of the form: (state, action, reward, next_state)
historical_data = [
    (0, 1, 0.5, 1),
    (1, 2, 1.0, 2),
    (2, 0, -0.5, 3),
    # ... add more data points
]

# Initialize Q-learning agent
# For demonstration, let's assume there are 4 states and 3 possible actions
agent = SS_RTB_QLearning(state_space_size=4, action_space_size=3)

# Training loop
for data_point in historical_data:
    state, action, reward, next_state = data_point
    agent.learn(state, action, reward, next_state)

# After training, you can reduce the exploration rate to make the agent act based on learned Q-values
agent.exploration_rate = 0.05  # or even set it to 0

# The Q-table after training on the sample data
print(agent.q_table)
