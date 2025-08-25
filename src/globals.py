GAME_SPEED = 7
WIDTH = 340
HEIGHT = 340
TILE_SIZE = 34
DIRECTIONS = {
    "UP": [0, -1],
    "DOWN": [0, 1],
    "LEFT": [-1, 0],
    "RIGHT": [1, 0],
}
LEARNING_RATE = 0.1  # Impact on the weights
DISCOUNT_FACTOR = 0.95  # Value of future rewards
EPSILON = 1  # Randomness in the agent's actions
EPSILON_DECAY = 0.9995  # Decay of randomness as the agent learns
MIN_EPSILON = 0.01  # Minimum randomness
MAX_EPISODES = 1  # Number of training episodes
MAX_STEPS = 10  # Maximum steps per episode
MEMORY_SIZE = 10000
BATCH_SIZE = 64  # Size of training batches
