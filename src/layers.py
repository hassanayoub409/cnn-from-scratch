"""Convolution, pooling and dense layers in plain NumPy (vectorized, with backpropagation).

Image batches have shape (m, height, width, channels).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pad_zeros(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))


def windows(X, f, stride):
    """All f x f windows of X as an array of shape (m, out_h, out_w, channels, f, f) (a view)."""
    return sliding_window_view(X, (f, f), axis=(1, 2))[:, ::stride, ::stride]


def conv_forward(X, W, b, stride=1, pad=0):
    """Convolve a batch X with filters W of shape (f, f, in_channels, out_channels)."""
    win = windows(pad_zeros(X, pad), W.shape[0], stride)
    Z = np.tensordot(win, W, axes=([4, 5, 3], [0, 1, 2])) + b
    return Z


def single_step(patch, W, b=0.0):
    """One filter applied to one patch: element-wise product, sum, add bias."""
    return float(np.sum(patch * W) + b)


def pool2d(x, f=2, stride=2, mode="max"):
    """Pool a single 2-D map. Returns an array of shape (out_h, out_w)."""
    win = sliding_window_view(x, (f, f))[::stride, ::stride]
    return win.max(axis=(-2, -1)) if mode == "max" else win.mean(axis=(-2, -1))


class Conv2D:
    def __init__(self, in_ch, out_ch, f, stride=1, pad=0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.f, self.stride, self.pad = f, stride, pad
        self.W = rng.normal(0, np.sqrt(2.0 / (f * f * in_ch)), (f, f, in_ch, out_ch))
        self.b = np.zeros(out_ch)

    def forward(self, X):
        self.win = windows(pad_zeros(X, self.pad), self.f, self.stride)
        self.x_shape = X.shape
        return np.tensordot(self.win, self.W, axes=([4, 5, 3], [0, 1, 2])) + self.b

    def backward(self, dZ):
        m, h, w, _ = dZ.shape
        f, s, p = self.f, self.stride, self.pad
        dW = np.tensordot(self.win, dZ, axes=([0, 1, 2], [0, 1, 2])).transpose(1, 2, 0, 3)
        db = dZ.sum(axis=(0, 1, 2))
        dXp = np.zeros((m, self.x_shape[1] + 2 * p, self.x_shape[2] + 2 * p, self.x_shape[3]))
        for i in range(f):  # each kernel offset scatters its share of the gradient back
            for j in range(f):
                dXp[:, i:i + h * s:s, j:j + w * s:s, :] += dZ @ self.W[i, j].T
        dX = dXp[:, p:dXp.shape[1] - p, p:dXp.shape[2] - p, :] if p else dXp
        return dX, {"W": dW, "b": db}


class MaxPool2D:
    def __init__(self, f=2, stride=2):
        self.f, self.stride = f, stride

    def forward(self, X):
        m, h, w, c = X.shape
        win = windows(X, self.f, self.stride)
        flat = win.reshape(*win.shape[:4], self.f * self.f)
        self.argmax = flat.argmax(axis=-1)
        self.x_shape = X.shape
        return flat.max(axis=-1)

    def backward(self, dA):
        _, h, w, _ = dA.shape
        s, f = self.stride, self.f
        dX = np.zeros(self.x_shape)
        for k in range(f * f):
            i, j = divmod(k, f)
            dX[:, i:i + h * s:s, j:j + w * s:s, :] += dA * (self.argmax == k)
        return dX


class Dense:
    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        return dZ @ self.W.T, {"W": self.X.T @ dZ, "b": dZ.sum(axis=0)}
