import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
epsilon = 0.5
decay = 0.995
min_epsilon = 0.01
episodes = 5000

# Track epsilon values
epsilons = []

for _ in range(episodes):
    epsilons.append(epsilon)
    epsilon = max(min_epsilon, epsilon * decay)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epsilons, label='Epsilon (exploration rate)')
plt.xlabel("Episode")
plt.ylabel("Epsilon Value")
plt.title("Epsilon Decay Over Episodes (Explore → Exploit)")
plt.grid(True)
plt.legend()
plt.show()
