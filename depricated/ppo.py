import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

# 1. Environment Setup
env = gym.make('CartPole-v1')  # Create the CartPole environment
state_dim = env.observation_space.shape[0]  # Dimension of state space
action_dim = env.action_space.n  # Number of actions


# 2. Model Definition
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Define the policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Define the value network
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        # Given a state, return the action probabilities and state value
        return self.policy(state), self.value(state)


# 3. PPO Algorithm
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, values, clip_epsilon=0.2):
    # Convert states to tensors
    states = [torch.FloatTensor(s) for s in states]

    # Now, stack the tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    returns = torch.stack(returns)
    values = torch.stack(values)

    # Calculate the advantages
    advantages = returns - values

    for _ in range(ppo_epochs):
        # Sample random mini-batches
        for start in range(0, len(states), mini_batch_size):
            end = start + mini_batch_size
            mini_batch_indices = np.random.randint(0, len(states), mini_batch_size)

            # Fetch mini-batch data
            mini_batch_states = states[mini_batch_indices]
            mini_batch_actions = actions[mini_batch_indices]
            mini_batch_log_probs = log_probs[mini_batch_indices]
            mini_batch_returns = returns[mini_batch_indices]
            mini_batch_advantages = advantages[mini_batch_indices]

            # Evaluate current policy and value function
            pi, v = model(mini_batch_states)
            pi = pi.gather(1, mini_batch_actions.unsqueeze(1)).squeeze(1)
            old_pi = torch.exp(mini_batch_log_probs)

            # Calculate the ratio
            ratio = pi / old_pi

            # Clipped surrogate objective
            clip_adv = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mini_batch_advantages
            policy_loss = -torch.min(ratio * mini_batch_advantages, clip_adv).mean()

            # Value function loss
            value_loss = 0.5 * (mini_batch_returns - v).pow(2).mean()

            # Update policy and value function
            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()


# Initialize the model and optimizer
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Hyperparameters
num_epochs = 1000
max_timesteps = 200
gamma = 0.99  # discount factor
gae_lambda = 0.95  # GAE lambda
ppo_epochs = 4
mini_batch_size = 64


# Function to compute returns using Generalized Advantage Estimation (GAE)
def compute_gae(next_value, rewards, masks, values, gamma=0.99, gae_lambda=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


for epoch in range(num_epochs):
    state = env.reset()[0]
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    episode_reward = 0

    for t in range(max_timesteps):
        # Ensure the state is a numpy array and then convert to tensor
        if isinstance(state, list):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = model(state_tensor)

        action = torch.multinomial(action_probs, 1).item()

        # Diagnostic print to inspect the output of env.step(action)
        step_output = env.step(action)

        next_state, reward, done, _, _ = step_output

        log_prob = torch.log(action_probs.squeeze(0)[action])
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        masks.append(1 - done)
        states.append(state)
        actions.append(torch.tensor(action))

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()[0]  # Reset the state if episode is done
        else:
            state = next_state

    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    with torch.no_grad():
        next_value = model(next_state)[1]
    returns = compute_gae(next_value, rewards, masks, values)

    # Update the model using PPO
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, values)

    print(f"Epoch: {epoch}, Reward: {episode_reward}")

