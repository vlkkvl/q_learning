import numpy as np
from communication.fulcrum import Fulcrum
from nn.reward import assign_reward


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
        # exploration
        if np.random.rand() < epsilon:
            return np.random.randint(self.nn.n_actions)

        # exploitation
        Q, _ = self.nn.forward(state[None, :])    # (1, A)
        return int(np.argmax(Q[0]))

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

            # reset environment
            self.env.reset()

            print(f"Episode {ep+1}/{num_episodes}")

            for t in range(max_steps):
                state = self.env.get_state()

                # choose action (epsilon-greedy)
                action = self.select_action(state, epsilon=self.epsilon)

                # perform action in maze
                next_state, moved, goal_reached = self.env.execute_action(action)

                # read sign
                sign = next_state[4:]

                # calculate reward in the next state
                reward, done = assign_reward(moved, goal_reached, sign)

                # Temporal Difference target
                Q_next, _ = self.nn.forward(next_state[None, :])  # value estimate of 1 step ahead
                target = reward if done else reward + self.gamma * np.max(Q_next[0])

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
