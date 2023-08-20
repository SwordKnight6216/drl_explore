import numpy as np

from agent import DQN
from env import SimpleEnvironment


def suggest_action(state):
    env = SimpleEnvironment()
    state_size = 1  # Price is the only state
    action_size = 3  # Three possible actions: increase, keep, or decrease the price
    dqn_agent = DQN(state_size, action_size)

    # NOTE: Load the trained model's weights if you've saved them previously
    dqn_agent.model.load_weights('models/trained_dqn_weights.h5')

    # Set exploration rate to 0 to always pick the best action
    dqn_agent.exploration_rate = 0

    state = np.reshape(state, [1, dqn_agent.state_size])

    # Ask the agent for the best action for this state
    best_action = dqn_agent.choose_action(state)

    # Display the suggested action
    action_map = {0: "Increase Price", 1: "Keep Price", 2: "Decrease Price"}
    return f"{action_map[best_action]}"
