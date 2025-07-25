import random
import gym  # pip install gym==0.25.2
import numpy as np

env = gym.make("Taxi-v3")
q_table = np.zeros((env.observation_space.n, env.action_space.n))
print("Q-table initialized with shape:", q_table, q_table.shape)
episodes = 10


def chhose_action(state):
    if np.random.rand() < 0.1:  # Epsilon-greedy policy
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state, :])  # Exploit


for episode in range(1, episodes + 1):
    state, _ = env.reset()
    print("state:", state)
    done = False

    while not done:  # try alternatively while True to see full fail
        action = chhose_action(state)


env.close()
