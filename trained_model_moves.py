import matplotlib.pyplot as plt
import numpy as np
import gym

try:
    

    # Environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode = 'human')  # non-slippery is easier to learn

    # Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    learning_rate = 0.8
    discount_factor = 0.99
    epsilon = 0.8  # Exploration - random action
    decay = 0.995 # slowly decreases
    min_epsilon = 0.01 # just 1% random  99% best action (exploit)
    episodes = 10000



    # Training
    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_episode_reward = 0

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
            total_episode_reward += reward

        rewards_per_episode.append(total_episode_reward)
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

    #testing part

    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(q_table[state])  # Exploit the learned Q-table
        state, reward, done, _, _ = env.step(action)
        print(f"Moved to state {state}, reward: {reward}")


    import time

    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)
        time.sleep(.01)  # Slow down for better viewing

except KeyboardInterrupt:
    print("\nTraining was interrupted. Plotting the progress so far...")

finally:
    # safe to plot even after Ctrl+C
    rolling_avg = np.convolve(rewards_per_episode, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title("Q-Learning: Avg Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward (last 100)")
    plt.grid(True)
    plt.show()
