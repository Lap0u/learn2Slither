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
        self.states = np.zeros((1200, len(globals.DIRECTIONS)))
        self.num_states = 1200
        self.num_actions = len(globals.DIRECTIONS)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        # print(f"Choosing action for state: {np.fliplr(np.rot90(state, 3))}")
        if np.random.rand() < globals.EPSILON:
            return np.random.choice(list(globals.DIRECTIONS.keys()))
        else:
            return list(globals.DIRECTIONS.keys())[np.argmax(self.q_table[state, :])]

    def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
        print("num", num_states, state)
        input_tensor = torch.zeros(num_states, dtype=torch.float32)
        conc_state = np.concatenate((state[0], state[1]))

        print("input", input_tensor, state)
        input_tensor[conc_state] = 1
        return input_tensor

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        # Convert states to one-hot encoded input tensors
        state_tensors = torch.stack(
            [self.state_to_dqn_input(s, self.num_states) for s in states]
        )
        next_state_tensors = torch.stack(
            [self.state_to_dqn_input(ns, self.num_states) for ns in next_states]
        )
        print("action", actions)
        action_tensors = torch.tensor(actions, dtype=torch.long)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32)
        done_tensors = torch.tensor(dones, dtype=torch.float32)

        # Current Q-values
        q_values = policy_dqn(state_tensors)
        q_values = q_values.gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        # Next Q-values (from target network, detached)
        next_q_values = target_dqn(next_state_tensors).max(1)[0].detach()
        target_q_values = reward_tensors + globals.GAMMA * next_q_values * (
            1 - done_tensors
        )

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the policy network
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
            while not done:
                if random.random() < globals.EPSILON:
                    action = np.random.choice(list(globals.DIRECTIONS.keys()))
                else:
                    action = list(globals.DIRECTIONS.keys())[
                        np.argmax(policy_dqn(game.snake_view).detach().numpy())
                    ]
                print("Action taken:", action)
                new_state, reward, done = game.step(action)
                if (done) or (step_count >= globals.MAX_STEPS):
                    sizes[episode] = game.snake.size
                memory.append((game.snake_view, action, new_state, reward, done))
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
