
import numpy as np
from agent import DQN
from env import SimpleEnvironment

# Instantiate the environment and DQN agent
env = SimpleEnvironment()
state_size = 1  # Price is the only state
action_size = 3  # Three possible actions: increase, keep, or decrease the price
dqn_agent = DQN(state_size, action_size)

# NOTE: Load the trained model's weights if you've saved them previously
dqn_agent.model.load_weights('models/trained_dqn_weights.h5')

# Set exploration rate to 0 to always pick the best action
dqn_agent.exploration_rate = 0

# Get the current state from the environment
state = env.reset()
state = np.reshape(state, [1, dqn_agent.state_size])

# Ask the agent for the best action for this state
best_action = dqn_agent.choose_action(state)

# Display the suggested action
action_map = {0: "Increase Price", 1: "Keep Price", 2: "Decrease Price"}
print(f"For the current state (Price: {env.current_price:.2f}), the suggested action is: {action_map[best_action]}")

synthetic_prices = [0.7, 1.1, 1.5, 1.9, 2.3]

for price in synthetic_prices:
    state = np.reshape([price], [1, state_size])

    # Ask the agent for the best action for this state
    best_action = dqn_agent.choose_action(state)

    # Display the suggested action
    action_map = {0: "Increase Price", 1: "Keep Price", 2: "Decrease Price"}
    print(f"For the synthetic price of {price:.2f}, the suggested action is: {action_map[best_action]}")
