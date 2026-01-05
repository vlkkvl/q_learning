import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Gradient of ReLU w.r.t. its input x.
    dout: upstream gradient (same shape as x)
    """
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx
