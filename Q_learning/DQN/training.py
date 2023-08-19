import numpy as np

from agent import DQN
from env import SimpleEnvironment

# Instantiate the environment and DQN agent
env = SimpleEnvironment()
state_size = 1  # Price is the only state
action_size = 3  # Three possible actions: increase, keep, or decrease the price
dqn_agent = DQN(state_size, action_size)

# Simulate interactions with the environment to train the DQN
EPISODES = 100
for episode in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = dqn_agent.choose_action(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        dqn_agent.remember(state, action, reward, next_state, done)
        state = next_state
        dqn_agent.learn(episode)
    if episode % 10 == 0:
        print(f"Episode {episode}, Price: {env.current_price:.2f}")

# Save the trained model's weights
dqn_agent.model.save_weights('models/trained_dqn_weights.h5')

# OPTIONAL: If you want to load the saved weights later
# dqn_agent.model.load_weights('trained_dqn_weights.h5')
