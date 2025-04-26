import gym
import numpy as np

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = 'human')  # non-slippery is easier to learn

# Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.5  # Exploration - random action
decay = 0.995 # slowly decreases
min_epsilon = 0.01 # just 1% random  99% best action (exploit)
episodes = 5000



# Training
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        new_state, reward, done, _, _ = env.step(action)

        # Q-learning formula
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

    epsilon = max(min_epsilon, epsilon * decay)

print("Training completed 🎉")

total_rewards = 0
for _ in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)
        total_rewards += reward

print(f"Success rate over 100 tries: {total_rewards}%")
