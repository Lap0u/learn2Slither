import src.globals as globals
import numpy as np
from src.game import Game
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import torch


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class AgentQ:
    def __init__(self):
        # Peut-etre enlever 2 et 2 pour les murs
        # self.states = np.zeros(
        #     (
        #         (
        #             globals.WIDTH
        #             // globals.TILE_SIZE
        #             * globals.HEIGHT
        #             // globals.TILE_SIZE
        #         )
        #         ** 4,  # Number of possible states (snake head + 3 apples)
        #         len(globals.DIRECTIONS),
        #     )
        # )
        # self.num_states = (
        #     globals.WIDTH // globals.TILE_SIZE * globals.HEIGHT // globals.TILE_SIZE
        # ) ** 4
        self.states = np.zeros((81, len(globals.DIRECTIONS)))
        self.num_states = 81
        self.num_actions = len(globals.DIRECTIONS)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        # print(f"Choosing action for state: {np.fliplr(np.rot90(state, 3))}")
        if np.random.rand() < globals.EPSILON:
            return np.random.choice(list(globals.DIRECTIONS.keys()))
        else:
            return list(globals.DIRECTIONS.keys())[np.argmax(self.q_table[state, :])]

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward
                        + globals.DISCOUNT_FACTOR
                        * target_dqn(
                            self.state_to_dqn_input(new_state, num_states)
                        ).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        memory = ReplayMemory(globals.MEMORY_SIZE)
        policy_dqn = DQN(
            in_states=self.num_states,
            h1_nodes=self.num_states,
            out_actions=self.num_actions,
        )
        target_dqn = DQN(
            in_states=self.num_states,
            h1_nodes=self.num_states,
            out_actions=self.num_actions,
        )
        sizes = np.zeros(globals.MAX_EPISODES)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=globals.LEARNING_RATE
        )
        rewards_per_episode = np.zeros(globals.MAX_EPISODES)
        epsilon_history = []
        step_count = 0
        for episode in range(globals.MAX_EPISODES):
            epsilon = globals.EPSILON
            game = Game()
            print("Starting environment\n", game.environment.T)
            done = False
            truncated = False
            while not done and not truncated:
                if random.random() < globals.EPSILON:
                    action = np.random.choice(list(globals.DIRECTIONS.keys()))
                else:
                    action = list(globals.DIRECTIONS.keys())[
                        np.argmax(policy_dqn(game.environment).detach().numpy())
                    ]
                print("Action taken:", action)
                new_state, reward, done = game.step(action)
                if (done) or (step_count >= globals.MAX_STEPS):
                    sizes[episode] = game.snake.size
                memory.append((game.environment, action, new_state, reward, done))
                step_count += 1
                rewards_per_episode[episode] = reward
                if len(memory) > globals.BATCH_SIZE:
                    print("Improving")
                    mini_batch = memory.sample(globals.BATCH_SIZE)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon - 1 / episode, globals.MIN_EPSILON)
                    epsilon_history.append(epsilon)
                    if step_count % 1000 == 0:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
        # torch.save(policy_dqn.state_dict(), "dqn_model.pth")
        print("Average snake size over episodes:", sizes.mean(), "Max:", sizes.max())
