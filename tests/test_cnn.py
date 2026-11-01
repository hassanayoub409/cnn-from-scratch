"""Run from the repo root:  python -m pytest -q"""
import numpy as np
import pytest

from src.layers import Conv2D, MaxPool2D, conv_forward, pad_zeros, pool2d, single_step
from src.model import DigitCNN
from src.transforms import shift


def naive_conv(X, W, b, stride, pad):
    """The textbook four-nested-loops definition of convolution."""
    Xp = pad_zeros(X, pad)
    m, H, Wd, _ = Xp.shape
    f, _, _, K = W.shape
    out_h, out_w = (H - f) // stride + 1, (Wd - f) // stride + 1
    Z = np.zeros((m, out_h, out_w, K))
    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for k in range(K):
                    patch = Xp[i, h * stride:h * stride + f, w * stride:w * stride + f, :]
                    Z[i, h, w, k] = single_step(patch, W[..., k], b[k])
    return Z


@pytest.mark.parametrize("stride,pad", [(1, 0), (2, 1), (1, 2), (2, 0)])
def test_vectorized_conv_matches_the_naive_loops(stride, pad):
    rng = np.random.default_rng(0)
    X, W, b = rng.normal(size=(3, 7, 6, 2)), rng.normal(size=(3, 3, 2, 4)), rng.normal(size=4)
    assert np.allclose(conv_forward(X, W, b, stride, pad), naive_conv(X, W, b, stride, pad))


def test_max_pool_forward_matches_manual_computation():
    x = np.arange(16.0).reshape(1, 4, 4, 1)
    out = MaxPool2D(2, 2).forward(x)
    assert np.allclose(out[0, :, :, 0], [[5, 7], [13, 15]])
    assert np.allclose(pool2d(x[0, :, :, 0], 2, 2, "average"), [[2.5, 4.5], [10.5, 12.5]])


def numeric_grad(f, arr, idx, eps=1e-6):
    old = arr[idx]
    arr[idx] = old + eps
    up = f()
    arr[idx] = old - eps
    down = f()
    arr[idx] = old
    return (up - down) / (2 * eps)


@pytest.mark.parametrize("stride,pad", [(1, 0), (2, 1)])
def test_conv_layer_gradients(stride, pad):
    rng = np.random.default_rng(1)
    layer = Conv2D(2, 3, f=3, stride=stride, pad=pad, rng=rng)
    X = rng.normal(size=(2, 6, 6, 2))
    target = rng.normal(size=layer.forward(X).shape)
    loss = lambda: float((layer.forward(X) * target).sum())
    layer.forward(X)
    dX, g = layer.backward(target)
    for name, arr, grad in (("W", layer.W, g["W"]), ("b", layer.b, g["b"]), ("X", X, dX)):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            assert np.isclose(grad[idx], numeric_grad(loss, arr, idx), rtol=1e-4, atol=1e-7), name


def test_full_model_gradients():
    rng = np.random.default_rng(2)
    model = DigitCNN(n_filters=3, seed=3)
    X, y = rng.uniform(size=(6, 8, 8, 1)), rng.integers(0, 10, 6)
    _, grads = model.loss_and_grads(X, y)
    loss = lambda: model.loss_and_grads(X, y)[0]
    for name, arr in model.params.items():
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            assert np.isclose(grads[name][idx], numeric_grad(loss, arr, idx), rtol=1e-3, atol=1e-7), name


def test_shift_moves_pixels_and_fills_with_zeros():
    img = np.zeros((1, 4, 4, 1))
    img[0, 1, 1, 0] = 1.0
    moved = shift(img, dx=1, dy=2)
    assert moved[0, 3, 2, 0] == 1.0 and moved.sum() == 1.0
    assert shift(img, dx=5, dy=0).sum() == 0.0
