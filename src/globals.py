GAME_SPEED = 1
ROBOT_SPEED = 7
WIDTH = 12
HEIGHT = 12
TILE_SIZE = 34
DIRECTIONS = {
    "UP": [[0, -1], 0],
    "DOWN": [[0, 1], 1],
    "LEFT": [[-1, 0], 2],
    "RIGHT": [[1, 0], 3],
}
LEARNING_RATE = 0.2  # Impact on the weights
DISCOUNT_FACTOR = 0.95  # Value of future rewards
EPSILON = 1  # Randomness in the agent's actions
EPSILON_DECAY = 0.998  # Decay of randomness as the agent learns
MIN_EPSILON = 0.02  # Minimum randomness
MAX_EPISODES = 3000  # Number of training episodes
MAX_STEPS = 1000  # Maximum steps per episode
MEMORY_SIZE = 10000
BATCH_SIZE = 32  # Size of training batches
