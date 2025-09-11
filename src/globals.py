GAME_SPEED = 6
ROBOT_SPEED = 50
WIDTH = 12
HEIGHT = 12
TILE = 34
DIR = {
    "UP": [[0, -1], 0],
    "DOWN": [[0, 1], 1],
    "LEFT": [[-1, 0], 2],
    "RIGHT": [[1, 0], 3],
}
LR = 0.001  # Learning rate
DISCOUNT_FACTOR = 0.95  # Value of future rewards
EPSILON = 1  # Randomness in the agent's actions
EPSILON_DECAY = 0.955  # Decay of randomness as the agent learns
MIN_EPSILON = 0.01  # Minimum randomness
MAX_STEPS = 1000  # Maximum steps per episode
MEMORY_SIZE = 30000
BATCH_SIZE = 32  # Size of training batches
PATIENCE = 50000
