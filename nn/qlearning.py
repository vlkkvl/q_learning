from collections import defaultdict, deque

import random
import numpy as np
from communication.fulcrum import Fulcrum
from diagnostics.full_diagnosis import diagnose_during_training
from nn.reward import assign_reward

class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


"""
Handles Q-Learning. Trains DPQ with experience replay 
with local Q-Learning for action selection & training
and target Q-Learning for bootstrapping the target.
"""
class QLearning:
    def __init__(self,
                 env: Fulcrum,
                 q_local,
                 q_target=None,
                 gamma=0.95,
                 epsilon=0.5,
                 epsilon_min=0.05,
                 epsilon_decay=0.995,
                 tau=0.005,
                 batch_size=32,
                 replay_capacity=10000,
                 learning_starts=100):  # start learning after this amount of steps
        self.env = env
        self.q_local = q_local
        self.q_target = q_target if q_target is not None else type(q_local)(
            n_features=q_local.n_features,
            hidden_size=q_local.hidden_size,
            n_actions=q_local.n_actions,
            learning_rate=q_local.lr,
        )

        # identical start
        self.q_target.copy_from(self.q_local)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        # experience replay
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.episode_losses = []
        self.total_steps = 0

    def valid_actions(self, state: np.ndarray) -> np.ndarray:
        available = state[:self.q_local.n_actions]
        valid = np.where(available > 0.5)[0]
        if valid.size == 0:
            return np.arange(self.q_local.n_actions)
        return valid

    def select_action(self, state, epsilon):
        """
        Choose an action using an ε-greedy policy.

        Parameters:
            state   (np.ndarray): Current state vector.
            epsilon (float):      Exploration rate. With probability ε a
                                 random action is chosen; otherwise the
                                 action with the highest predicted Q-value.

        Returns:
            int: Index of the selected action in [0, n_actions-1].
        """
        valid_actions = self.valid_actions(state)
        # exploration
        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_actions))

        # exploitation
        Q, _ = self.q_local.forward(state[None, :])  # (1, A)
        q_values = Q[0].copy()

        # mask invalid actions
        invalid_mask = np.ones(self.q_local.n_actions, dtype=bool)
        invalid_mask[valid_actions] = False
        q_values[invalid_mask] = -np.inf
        return int(np.argmax(q_values))

    def compute_targets(self, rewards, next_states, dones):
        """Compute TD targets for a batch."""
        Q_next, _ = self.q_target.forward(next_states)

        # mask invalid actions for each next_state
        targets = np.zeros(len(rewards))
        for i in range(len(rewards)):
            if dones[i]:
                targets[i] = rewards[i]
            else:  # compute targets where goal was not reached
                next_valid = self.valid_actions(next_states[i])
                q_vals = Q_next[i].copy()
                invalid_mask = np.ones(self.q_local.n_actions, dtype=bool)
                invalid_mask[next_valid] = False
                q_vals[invalid_mask] = -np.inf
                targets[i] = rewards[i] + self.gamma * np.max(q_vals)

        return targets

    def td_step(self, states, actions, targets):
        """
        Perform one TD(0) / Q-learning update on a minibatch.

        Parameters:
            - states  (np.ndarray): shape (N, n_features)
                Batch of input states.
            - actions (np.ndarray): shape (N, )
                Action index taken in each state.
            - targets (np.ndarray): shape (N, )
                TD targets: r + gamma * max_a' Q(s',a'),
                or just r for terminal transitions.

        Returns:
            float: Mean squared TD error for the minibatch.
        """
        Q, cache = self.q_local.forward(states)  # (N, A)
        N = Q.shape[0]

        # predicted Q for taken actions
        q_sa = Q[np.arange(N), actions]  # (N,)

        # MSE loss
        loss = 0.5 * np.mean((q_sa - targets) ** 2)

        # gradient dL/dQ
        dQ = np.zeros_like(Q)
        dQ[np.arange(N), actions] = (q_sa - targets) / N  # d/dq_sa of MSE

        grads = self.q_local.backward(cache, dQ)
        self.q_local.update(grads)

        # soft update target after updating local
        self.q_target.soft_update_from(self.q_local, tau=self.tau)

        return loss

    def train(self, num_episodes: int, max_steps: int):
        """
        Run Q-learning for num_episodes with experience replay.
        Each episode:
         - reset env
         - interact up to max_steps or until 'done'
        """
        for ep in range(num_episodes):
            episode_loss = 0.0
            episode_reward = 0.0
            steps = 0

            # reset environment
            self.env.reset()

            visited_counts = defaultdict(int)  # (x,y) -> number of visits in THIS episode
            history = deque(maxlen=6)

            current_position = self.env.get_coordinates()
            visited_counts[current_position] += 1
            history.append(current_position)
            goal_position = self.env.get_goal_coordinates()

            for t in range(max_steps):
                state = self.env.get_state()

                # choose action (epsilon-greedy)
                action = self.select_action(state, epsilon=self.epsilon)

                # perform action in maze
                next_state, moved, goal_reached = self.env.execute_action(action)
                next_position = self.env.get_coordinates()

                # read signs
                current_sign = state[4:12]  # for 'prefer direction' signs
                next_sign = next_state[4:12]  # for dead ends

                # calculate reward in the next state
                reward, done, details = assign_reward(
                    moved,
                    goal_reached,
                    next_sign,
                    current_sign,
                    action,
                    next_state[:self.q_local.n_actions],  # next available actions
                    current_position,
                    next_position,
                    visited_counts,
                    goal_position,
                )
                # update tracking
                if moved:
                    current_position = next_position
                    visited_counts[current_position] += 1
                    history.append(current_position)

                diagnose_during_training(
                    self,
                    ep,
                    state,
                    action,
                    reward,
                    details,
                    next_state,
                    current_sign,
                    next_sign,
                    self.epsilon,
                    done
                )

                # store transition in replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                episode_reward += reward
                self.total_steps += 1

                # learn from replay buffer
                if len(self.replay_buffer) >= self.learning_starts:
                    # sample batch
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                        self.replay_buffer.sample(self.batch_size)

                    # compute targets
                    targets = self.compute_targets(batch_rewards, batch_next_states, batch_dones)

                    # update Q-network
                    loss = self.td_step(batch_states, batch_actions, targets)

                    episode_loss += loss
                    steps += 1

                if done:  # goal reached
                    print(f"Episode {ep + 1}: GOAL REACHED in {t + 1} steps")
                    break

            # TD error in each episode
            avg_loss = episode_loss / max(steps, 1)
            self.episode_losses.append(avg_loss)

            # decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def test(self, num_episodes=20, max_steps=100):
        """
        Evaluate the learned policy:
        - no exploration (pure greedy)
        - count episodes where goal is reached.
        """
        successes = 0
        for ep in range(num_episodes):
            self.env.reset()
            print(f"Episode {ep + 1}/{num_episodes}")
            for t in range(max_steps):
                state = self.env.get_state()
                # greedy policy during test
                action = self.select_action(state, epsilon=0.0)
                _, moved, goal_reached = self.env.execute_action(action)
                if goal_reached:
                    successes += 1
                    break

        print(f"Successes: {successes}/{num_episodes}")