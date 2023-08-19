import tensorflow as tf
import numpy as np
from collections import deque
import random
class DQN:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.9, exploration_rate=1.0,
                 exploration_decay=0.995, batch_size=64, memory_size=2000):
        # Size of the state and action spaces
        self.state_size = state_size
        self.action_size = action_size

        # Create the main neural network model and its target counterpart
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)

        # Initialize the target model with the same weights as the main model
        self.update_target_model()

        # Factor to discount future rewards
        self.discount_factor = discount_factor

        # Parameters governing the exploration vs exploitation trade-off
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay

        # Experience replay buffer to store experiences and batch size for training
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def _build_model(self, learning_rate):
        # Define a feedforward neural network
        model = tf.keras.models.Sequential()

        # First hidden layer with 24 neurons and ReLU activation
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))

        # Second hidden layer with 24 neurons and ReLU activation
        model.add(tf.keras.layers.Dense(24, activation='relu'))

        # Output layer with a neuron for each action, linear activation
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))

        # Compile the model using Mean Squared Error loss and Adam optimizer
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        return model

    def update_target_model(self):
        # Copy the weights from the main model to the target model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in the replay memory
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # Decide to explore or exploit based on exploration rate
        if np.random.rand() <= self.exploration_rate:
            # Explore: Choose a random action
            return random.randrange(self.action_size)

        # Exploit: Choose the action with highest predicted Q-value
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self):
        # Ensure there are enough samples in memory to form a batch
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch from the memory
        batch = random.sample(self.memory, self.batch_size)

        # Extract states and next states from the batch to predict Q-values in batch
        states = np.array([i[0] for i in batch])
        next_states = np.array([i[3] for i in batch])

        # Predict Q-values for the current states and next states
        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(next_states)

        # Update the Q-values based on the received rewards and the max Q-values of next states
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                q_values[i][action] = reward
            else:
                q_values[i][action] = reward + self.discount_factor * np.amax(q_values_next[i])

        # Train the neural network with the states as inputs and updated Q-values as targets
        self.model.fit(states, q_values, verbose=0)

        # Decay the exploration rate to gradually shift from exploration to exploitation
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

# Comments have been added to each line or block to explain the code's functionality.


# Define the environment

class SimpleEnvironment:
    def __init__(self, initial_price=1.0):
        self.current_price = initial_price

    def step(self, action):
        # Increase, keep the same, or decrease the price based on the action
        if action == 0:
            self.current_price *= 1.1
        elif action == 2:
            self.current_price *= 0.9

        # Here, reward is the new price (this is a simple and naive reward function for demonstration)
        reward = self.current_price

        # Next state is the new price
        next_state = [self.current_price]

        # Simple condition to check if the episode is done (for demonstration purposes)
        done = True if self.current_price > 2.0 or self.current_price < 0.5 else False

        return next_state, reward, done

    def reset(self):
        self.current_price = 1.0
        return [self.current_price]


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
        dqn_agent.learn()
    if episode % 10 == 0:
        print(f"Episode {episode}, Price: {env.current_price:.2f}")

# Test results after 100 episodes of training
