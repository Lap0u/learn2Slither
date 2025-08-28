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
        self.fc2 = nn.Linear(h1_nodes, h1_nodes // 2)
        self.fc3 = nn.Linear(h1_nodes // 2, h1_nodes // 4)
        self.out = nn.Linear(h1_nodes // 4, out_actions)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
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
        # Cross-view: WIDTH + HEIGHT values + 4 direction encoding
        self.state_size = globals.WIDTH + globals.HEIGHT + 4
        self.num_actions = len(globals.DIRECTIONS)
        self.loss_fn = nn.MSELoss()

    def get_direction_encoding(self, direction):
        """Convert direction string to one-hot encoding"""
        encoding = [0, 0, 0, 0]
        direction_to_idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        if direction in direction_to_idx:
            encoding[direction_to_idx[direction]] = 1
        return encoding

    def state_to_dqn_input(self, snake_view, current_direction="RIGHT") -> torch.Tensor:
        """
        Convert snake's cross-view to DQN input tensor.
        snake_view is a tuple of (x_view, y_view) where:
        - x_view: what the snake sees in its column (length WIDTH)
        - y_view: what the snake sees in its row (length HEIGHT)
        """
        if isinstance(snake_view, tuple) and len(snake_view) == 2:
            x_view, y_view = snake_view

            # Normalize the view values (0-5 range to 0-1 range)
            x_view_norm = np.array(x_view, dtype=np.float32) / 5.0
            y_view_norm = np.array(y_view, dtype=np.float32) / 5.0

            # Get direction encoding
            direction_encoding = self.get_direction_encoding(current_direction)

            # Combine cross-view and direction
            full_state = np.concatenate([x_view_norm, y_view_norm, direction_encoding])

        else:
            # Fallback for invalid state
            full_state = np.zeros(self.state_size)

        return torch.tensor(full_state, dtype=torch.float32)

    def choose_action(self, snake_view, policy_dqn, epsilon, current_direction="RIGHT"):
        """Choose action using epsilon-greedy policy with cross-view only"""
        if random.random() < epsilon:
            return random.choice(list(globals.DIRECTIONS.keys()))
        else:
            with torch.no_grad():
                state_tensor = self.state_to_dqn_input(snake_view, current_direction)
                q_values = policy_dqn(state_tensor.unsqueeze(0))
                action_idx = q_values.argmax().item()
                return list(globals.DIRECTIONS.keys())[action_idx]

    def calculate_distance_reward(self, x_view, y_view, head_x, head_y):
        """
        Calculate a simple distance-based reward using only cross-view information.
        Look for apples (value 2) in the cross-view.
        """
        reward = 0

        # Check x_view for apples and calculate distance
        for i, cell_value in enumerate(x_view):
            if cell_value == 2:  # Green apple
                distance = abs(i - head_y)  # Distance in y direction
                reward += max(0, 2 - distance * 0.1)  # Closer = better reward

        # Check y_view for apples and calculate distance
        for i, cell_value in enumerate(y_view):
            if cell_value == 2:  # Green apple
                distance = abs(i - head_x)  # Distance in x direction
                reward += max(0, 2 - distance * 0.1)  # Closer = better reward

        return reward * 0.1  # Scale down the reward

    def get_position_from_view(self, snake_view):
        """Extract snake head position from cross-view (where both views show snake head)"""
        if isinstance(snake_view, tuple) and len(snake_view) == 2:
            x_view, y_view = snake_view

            # Find snake head (value 4) in both views
            head_y = None
            head_x = None

            for i, val in enumerate(x_view):
                if val == 4:  # Snake head
                    head_y = i
                    break

            for i, val in enumerate(y_view):
                if val == 4:  # Snake head
                    head_x = i
                    break

            return head_x, head_y
        return None, None

    def get_shaped_reward(self, old_view, new_view, base_reward):
        """Add reward shaping using only cross-view information"""
        shaped_reward = base_reward

        # Only add distance-based rewards for non-terminal states
        if base_reward > -50:  # Not a death
            old_head_x, old_head_y = self.get_position_from_view(old_view)
            new_head_x, new_head_y = self.get_position_from_view(new_view)

            if old_head_x is not None and new_head_x is not None:
                old_x_view, old_y_view = old_view
                new_x_view, new_y_view = new_view

                old_dist_reward = self.calculate_distance_reward(
                    old_x_view, old_y_view, old_head_x, old_head_y
                )
                new_dist_reward = self.calculate_distance_reward(
                    new_x_view, new_y_view, new_head_x, new_head_y
                )

                # Small reward for getting closer to apples
                shaped_reward += new_dist_reward - old_dist_reward

        return shaped_reward

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones, directions, next_directions = zip(
            *mini_batch
        )

        # Convert states to input tensors
        state_tensors = torch.stack(
            [self.state_to_dqn_input(s, d) for s, d in zip(states, directions)]
        )
        next_state_tensors = torch.stack(
            [
                self.state_to_dqn_input(ns, nd)
                for ns, nd in zip(next_states, next_directions)
            ]
        )

        action_tensors = torch.tensor(actions, dtype=torch.long)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32)
        done_tensors = torch.tensor(dones, dtype=torch.bool)

        # Current Q-values for the taken actions
        q_values = policy_dqn(state_tensors)
        q_values = q_values.gather(1, action_tensors.unsqueeze(1)).squeeze(1)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = target_dqn(next_state_tensors).max(1)[0]
            target_q_values = (
                reward_tensors
                + globals.DISCOUNT_FACTOR * next_q_values * (~done_tensors).float()
            )

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self):
        memory = ReplayMemory(globals.MEMORY_SIZE)
        policy_dqn = DQN(
            in_states=self.state_size,
            h1_nodes=256,  # Smaller network for simpler state space
            out_actions=self.num_actions,
        )
        target_dqn = DQN(
            in_states=self.state_size,
            h1_nodes=256,
            out_actions=self.num_actions,
        )

        # Initialize networks
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)

        # Training tracking
        sizes = np.zeros(globals.MAX_EPISODES)
        rewards_per_episode = np.zeros(globals.MAX_EPISODES)
        epsilon_history = []
        losses = []
        step_count = 0

        current_epsilon = 0.9  # Start with high exploration

        for episode in range(globals.MAX_EPISODES):
            print(f"Starting episode {episode + 1}/{globals.MAX_EPISODES}")
            game = Game()
            episode_reward = 0
            episode_steps = 0
            done = False

            # Store previous state for reward shaping
            prev_snake_view = game.snake_view

            while not done and episode_steps < globals.MAX_STEPS:
                current_direction = game.snake.direction

                # Choose action using cross-view only
                action = self.choose_action(
                    game.snake_view, policy_dqn, current_epsilon, current_direction
                )

                # Convert action to index for storage
                action_idx = list(globals.DIRECTIONS.keys()).index(action)

                # Store the current state before taking action
                old_snake_view = game.snake_view
                old_direction = current_direction

                # Take action and observe result
                new_snake_view, base_reward, done = game.step(action)

                # Apply reward shaping using only cross-view information
                reward = self.get_shaped_reward(
                    old_snake_view, new_snake_view, base_reward
                )

                # Improved reward structure
                if base_reward == -100:  # Death
                    reward = -100
                elif base_reward == 10:  # Ate apple
                    reward = 50  # Higher reward for eating
                elif base_reward == -10:  # Ate red apple
                    reward = -20
                else:  # Normal step
                    reward = -0.1  # Small penalty for time to encourage efficiency

                # Store transition in memory
                memory.append(
                    (
                        old_snake_view,
                        action_idx,
                        new_snake_view,
                        reward,
                        done,
                        old_direction,
                        game.snake.direction,
                    )
                )

                episode_reward += reward
                episode_steps += 1
                step_count += 1

                # Train the network if we have enough samples
                if len(memory) >= globals.BATCH_SIZE:
                    mini_batch = memory.sample(globals.BATCH_SIZE)
                    loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                    losses.append(loss)

                    # Decay epsilon more gradually for cross-view learning
                    current_epsilon = max(current_epsilon * 0.9998, 0.02)

                    # Update target network
                    if step_count % 1500 == 0:  # Less frequent updates
                        target_dqn.load_state_dict(policy_dqn.state_dict())

            # Record episode statistics
            sizes[episode] = game.snake.size
            rewards_per_episode[episode] = episode_reward
            epsilon_history.append(current_epsilon)

            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(
                    rewards_per_episode[max(0, episode - 100) : episode + 1]
                )
                avg_size = np.mean(sizes[max(0, episode - 100) : episode + 1])
                avg_loss = np.mean(losses[-100:]) if losses else 0
                print(
                    f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                    f"Avg Size: {avg_size:.2f}, Epsilon: {current_epsilon:.4f}, "
                    f"Avg Loss: {avg_loss:.4f}, Steps: {episode_steps}"
                )

        # Save the trained model
        torch.save(policy_dqn.state_dict(), "dqn_snake_model.pth")
        print("Training completed!")
        print(f"Average snake size: {sizes.mean():.2f}, Max: {sizes.max()}")
        print(f"Final epsilon: {current_epsilon:.4f}")

        return {
            "sizes": sizes,
            "rewards": rewards_per_episode,
            "epsilon_history": epsilon_history,
            "losses": losses,
            "policy_dqn": policy_dqn,
        }
