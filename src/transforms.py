"""Simple image perturbations used to test how robust a model is."""
import numpy as np


def shift(X, dx, dy):
    """Move every image dx pixels right and dy pixels down; vacated pixels become 0."""
    out = np.zeros_like(X)
    h, w = X.shape[1:3]
    if abs(dx) >= w or abs(dy) >= h:  # everything is pushed out of the frame
        return out
    src_y, dst_y = slice(max(0, -dy), min(h, h - dy)), slice(max(0, dy), min(h, h + dy))
    src_x, dst_x = slice(max(0, -dx), min(w, w - dx)), slice(max(0, dx), min(w, w + dx))
    out[:, dst_y, dst_x] = X[:, src_y, src_x]
    return out


def add_noise(X, sigma, rng):
    return np.clip(X + rng.normal(0, sigma, X.shape), 0.0, 1.0) if sigma > 0 else X.copy()


def random_shift(X, amount, rng):
    """Shift each image by `amount` pixels in one random direction (up, down, left or right)."""
    out = X.copy()
    if amount == 0:
        return out
    directions = rng.integers(0, 4, len(X))
    for k, (dx, dy) in enumerate([(amount, 0), (-amount, 0), (0, amount), (0, -amount)]):
        idx = directions == k
        out[idx] = shift(X[idx], dx, dy)
    return out


def augment_batch(X, rng, max_shift=1):
    """Training-time augmentation: every image gets a random shift of up to max_shift pixels."""
    out = np.empty_like(X)
    dxs = rng.integers(-max_shift, max_shift + 1, len(X))
    dys = rng.integers(-max_shift, max_shift + 1, len(X))
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            idx = (dxs == dx) & (dys == dy)
            if idx.any():
                out[idx] = shift(X[idx], dx, dy)
    return out
