import numpy as np
from communication.fulcrum import Fulcrum
from nn.reward import assign_reward
from simulation.signs import SignType


class QLearning:
    def __init__(self, env: Fulcrum, network, gamma=0.95,
                 epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.995):
        self.nn = network
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon  # exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode_losses = []

    def valid_actions(self, state: np.ndarray) -> np.ndarray:
        available = state[:self.nn.n_actions]
        valid = np.where(available > 0.5)[0]
        if valid.size == 0:
            return np.arange(self.nn.n_actions)
        return valid

    def select_action(self, state, epsilon):
        """
        Choose an action using an ε-greedy policy.

        Parameters:
            state   (np.ndarray): Current state vector (n_features,).
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
        Q, _ = self.nn.forward(state[None, :])    # (1, A)
        q_values = Q[0].copy()
        invalid_mask = np.ones(self.nn.n_actions, dtype=bool)
        invalid_mask[valid_actions] = False
        q_values[invalid_mask] = -np.inf
        return int(np.argmax(q_values))

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
        Q, cache = self.nn.forward(states)  # (N, A)
        N = Q.shape[0]

        # predicted Q for taken actions
        q_sa = Q[np.arange(N), actions]  # (N,)

        # MSE loss
        loss = 0.5 * np.mean((q_sa - targets) ** 2)

        # gradient dL/dQ
        dQ = np.zeros_like(Q)
        dQ[np.arange(N), actions] = (q_sa - targets) / N  # d/dq_sa of MSE

        grads = self.nn.backward(cache, dQ)
        self.nn.update(grads)
        return loss

    def train(self, num_episodes: int, max_steps: int):
        """
        Run Q-learning for num_episodes.
        Each episode:
         - reset env
         - interact up to max_steps or until 'done'
        """
        for ep in range(num_episodes):
            total_loss = 0.0
            steps = 0
            visited_positions = set()
            visit_counts = {}
            previous_position = None
            previous_distance = None

            # reset environment
            self.env.reset()
            current_position = self.env.get_coordinates()
            visited_positions.add(current_position)
            visit_counts[current_position] = 1
            goal_position = self.env.get_goal_coordinates()
            if goal_position is not None:
                dx = abs(current_position[0] - goal_position[0])
                dy = abs(current_position[1] - goal_position[1])
                previous_distance = dx + dy

            print(f"Episode {ep+1}/{num_episodes}")

            for t in range(max_steps):
                state = self.env.get_state()
                if state.shape[0] != self.nn.n_features:
                    raise ValueError(
                        f"State size {state.shape[0]} does not match network input size "
                        f"{self.nn.n_features}."
                    )

                # choose action (epsilon-greedy)
                action = self.select_action(state, epsilon=self.epsilon)

                # perform action in maze
                next_state, moved, goal_reached = self.env.execute_action(action)

                # read sign
                sign = next_state[4:]

                # calculate reward in the next state
                reward, done = assign_reward(moved, goal_reached, sign)

                if moved:
                    position = self.env.get_coordinates()
                    if position not in visited_positions:
                        reward += 0.1 # progress bonus
                        visited_positions.add(position)
                    else:
                        reward -= 0.1 # revisit penalty
                    visit_counts[position] = visit_counts.get(position, 0) + 1
                    if (
                            previous_position is not None
                            and position == previous_position
                            and SignType.from_vector(sign) != SignType.DEAD_END
                    ):
                        reward -= 0.5 # backtrack penalty
                    if goal_position is not None:
                        dx = abs(position[0] - goal_position[0])
                        dy = abs(position[1] - goal_position[1])
                        distance = dx + dy
                        if previous_distance is not None and distance < previous_distance:
                            reward += 0.05 # distance_bonus
                        previous_distance = distance
                    previous_position = current_position
                    current_position = position

                # Temporal Difference target
                Q_next, _ = self.nn.forward(next_state[None, :])  # value estimate of 1 step ahead
                next_q_values = Q_next[0].copy()
                next_valid_actions = self.valid_actions(next_state)
                next_invalid_mask = np.ones(self.nn.n_actions, dtype=bool)
                next_invalid_mask[next_valid_actions] = False
                next_q_values[next_invalid_mask] = -np.inf
                target = reward if done else reward + self.gamma * np.max(next_q_values)

                if done:
                    print("Goal reached.")

                # one gradient step
                loss = self.td_step(
                    states=state[None, :],
                    actions=np.array([action]),
                    targets=np.array([target]),
                )

                total_loss += loss
                steps += 1

                if done:
                    break

            print("Avg TD Loss:", total_loss / steps)

            # TD error over all episodes
            self.episode_losses.append(total_loss / steps)

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
            print(f"Episode {ep+1}/{num_episodes}")
            for t in range(max_steps):
                state = self.env.get_state()
                # greedy policy during test
                action = self.select_action(state, epsilon=0.0)
                _, moved, goal_reached = self.env.execute_action(action)
                if goal_reached:
                    successes += 1
                    break

        print(f"Successes: {successes}/{num_episodes}")
