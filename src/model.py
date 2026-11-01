"""A small CNN for 8x8 digits:  CONV(3x3) -> ReLU -> MAXPOOL(2x2) -> FLATTEN -> DENSE(10) -> softmax."""
import numpy as np

from .layers import Conv2D, Dense, MaxPool2D
from .optim import Adam


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class DigitCNN:
    def __init__(self, n_filters=8, image_size=8, n_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        self.conv = Conv2D(1, n_filters, f=3, rng=rng)
        self.pool = MaxPool2D(2, 2)
        pooled = (image_size - 2) // 2
        self.dense = Dense(pooled * pooled * n_filters, n_classes, rng)
        self.n_filters = n_filters
        self.params = {"conv_W": self.conv.W, "conv_b": self.conv.b,
                       "dense_W": self.dense.W, "dense_b": self.dense.b}

    # ----- forward -----
    def forward(self, X):
        """Returns (logits, intermediate activations)."""
        z = self.conv.forward(X)
        a = np.maximum(z, 0.0)
        p = self.pool.forward(a)
        self._relu_mask, self._pool_shape = z > 0, p.shape
        logits = self.dense.forward(p.reshape(len(X), -1))
        return logits, {"conv": z, "relu": a, "pool": p}

    def predict_proba(self, X):
        return softmax(self.forward(X)[0])

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    # ----- training -----
    def loss_and_grads(self, X, y):
        logits, _ = self.forward(X)
        probs = softmax(logits)
        m = len(y)
        loss = -np.mean(np.log(probs[np.arange(m), y] + 1e-12))
        d = probs.copy()
        d[np.arange(m), y] -= 1.0
        d /= m
        d_flat, g_dense = self.dense.backward(d)
        d_relu = self.pool.backward(d_flat.reshape(self._pool_shape))
        _, g_conv = self.conv.backward(d_relu * self._relu_mask)
        return loss, {"conv_W": g_conv["W"], "conv_b": g_conv["b"],
                      "dense_W": g_dense["W"], "dense_b": g_dense["b"]}

    def fit(self, X, y, X_val, y_val, epochs=60, batch_size=32, lr=0.003, augment=None,
            patience=15, seed=0, log=None):
        """Mini-batch Adam. `augment(X_batch, rng)` is applied to every training batch if given."""
        rng = np.random.default_rng(seed)
        opt = Adam(self.params, lr=lr)
        best_acc, best, best_epoch = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                b = order[start:start + batch_size]
                xb = augment(X[b], rng) if augment else X[b]
                _, grads = self.loss_and_grads(xb, y[b])
                opt.step(grads)
            val_acc = float(np.mean(self.predict(X_val) == y_val))
            if log and epoch % 5 == 0:
                log(f"  epoch {epoch:3d}  val accuracy {val_acc:.3f}")
            if val_acc >= best_acc:  # ties go to the later epoch
                best_acc, best_epoch = val_acc, epoch
                best = {k: v.copy() for k, v in self.params.items()}
            elif epoch - best_epoch >= patience:
                break
        for k, v in best.items():
            self.params[k][...] = v
        return best_acc, best_epoch

    # ----- persistence -----
    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(n_filters=data["conv_W"].shape[-1])
        for k in model.params:
            model.params[k][...] = data[k]
        return model
