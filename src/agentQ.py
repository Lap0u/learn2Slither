import src.globals as globals
import numpy as np
from src.game import Game


class AgentQ:
    def __init__(self):
        # Peut-etre enlever 2 et 2 pour les murs
        self.q_table = np.zeros(
            (
                (
                    globals.WIDTH
                    // globals.TILE_SIZE
                    * globals.HEIGHT
                    // globals.TILE_SIZE
                )
                ** 4,  # Number of possible states (snake head + 3 apples)
                len(globals.DIRECTIONS),
            )
        )
        print("Q-table initialized with shape:", self.q_table.shape)

    def choose_action(self, state):
        # print(f"Choosing action for state: {np.fliplr(np.rot90(state, 3))}")
        if np.random.rand() < globals.EPSILON:
            return np.random.choice(list(globals.DIRECTIONS.keys()))
        else:
            return list(globals.DIRECTIONS.keys())[np.argmax(self.q_table[state, :])]

    def train(self):
        for episode in range(globals.MAX_EPISODES):
            game = Game()
            done = False
            for step in range(globals.MAX_STEPS):
                action = self.choose_action(game.environment)
                print("Action chosen:", action)
                old_state = game.environment.copy()
                # print(f"Episode {episode}, Step {step}, Action: {action}")
                new_state, reward, is_dead = game.step(action)
                # print(f"Reward: {reward}, Is Dead: {is_dead}\n\n")
                print(self.q_table.shape, self.q_table)
                old_value = self.q_table[old_state, globals.DIRECTIONS[action]]
                # old_value = self.q_table[old_state, globals.DIRECTIONS[action]]
