import numpy as np
from nn.activation import relu, relu_backward

class NeuralNetwork:
    def __init__(
        self,
        n_features: int = 16, # l_free, r_free, up_free, down_free, [rule], x_norm, y_norm, goal_x_norm, goal_y_norm
        hidden_size: int = 32,
        n_actions: int = 4,      # left, right, up, down
        learning_rate: float = 0.001,
        weight_scale: float = 1.0
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.lr = learning_rate

        self.W1 = np.random.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features) * weight_scale
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, n_actions) * np.sqrt(2.0 / hidden_size) * weight_scale
        self.b2 = np.zeros(n_actions)

    def forward(self, X: np.ndarray):
        """
        X: (N, n_features)
        Returns:
            Q: (N, n_actions) – Q(s,a) for each action
        """
        z1 = X @ self.W1 + self.b1      # (N, H)
        a1 = relu(z1)                   # (N, H)
        z2 = a1 @ self.W2 + self.b2     # (N, A) -> Q-values
        cache = {"X": X, "z1": z1, "a1": a1}
        return z2, cache

    def backward(self, cache, dQ: np.ndarray):
        """
        dQ: dL/dQ (same shape as Q: (N, n_actions))
        """
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]

        # output layer
        dW2 = a1.T @ dQ                 # (H, A)
        db2 = np.sum(dQ, axis=0)        # (A,)

        # backprop into hidden
        dA1 = dQ @ self.W2.T            # (N, H)
        dZ1 = relu_backward(dA1, z1)    # (N, H)

        dW1 = X.T @ dZ1                 # (F, H)
        db1 = np.sum(dZ1, axis=0)       # (H,)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads

    def update(self, grads):
        self.W1 -= self.lr * grads["W1"]
        self.b1 -= self.lr * grads["b1"]
        self.W2 -= self.lr * grads["W2"]
        self.b2 -= self.lr * grads["b2"]

    def copy_from(self, other: "NeuralNetwork"):
        """Hard copy: self <- other"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def soft_update_from(self, other: "NeuralNetwork", tau: float):
        """
        Soft update: self <- tau*other + (1-tau)*self.
        self is target, other is local.
        """
        self.W1 = tau * other.W1 + (1.0 - tau) * self.W1
        self.b1 = tau * other.b1 + (1.0 - tau) * self.b1
        self.W2 = tau * other.W2 + (1.0 - tau) * self.W2
        self.b2 = tau * other.b2 + (1.0 - tau) * self.b2