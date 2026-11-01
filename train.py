"""Train and evaluate the CNNs on the scikit-learn digits dataset.

    python train.py        (needs: pip install -r requirements-train.txt)

Trains two variants (plain, and with shift augmentation) with 5 random seeds each, plus a
logistic-regression baseline, and measures accuracy on clean, shifted and noisy test images.
The seed-0 models are saved for the app.
"""
import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.model import DigitCNN
from src.paths import DATA_DIR, MODELS_DIR
from src.transforms import add_noise, augment_batch, random_shift

N_FILTERS, N_SEEDS = 8, 5
CONDITIONS = {  # name -> function turning clean test images into perturbed ones
    "clean": lambda X: X,
    "shift 1px": lambda X: random_shift(X, 1, np.random.default_rng(1)),
    "shift 2px": lambda X: random_shift(X, 2, np.random.default_rng(1)),
    "noise 0.1": lambda X: add_noise(X, 0.1, np.random.default_rng(2)),
    "noise 0.2": lambda X: add_noise(X, 0.2, np.random.default_rng(2)),
    "noise 0.3": lambda X: add_noise(X, 0.3, np.random.default_rng(2)),
}


def accuracy_table(predict, X_test, y_test):
    return {name: float(np.mean(predict(make(X_test)) == y_test)) for name, make in CONDITIONS.items()}


def main():
    digits = load_digits()
    X, y = (digits.images / 16.0)[..., None], digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=0)
    print(f"{len(y_train)} train / {len(y_val)} validation / {len(y_test)} test images (8x8)")

    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATA_DIR / "digits_test.npz",
                        X=np.rint(X_test[..., 0] * 16).astype(np.uint8), y=y_test)

    per_seed = {"Logistic regression": [], "CNN": [], "CNN + shift augmentation": []}
    shipped, confusion = {}, {}

    flat = lambda A: A.reshape(len(A), -1)
    lr = LogisticRegression(max_iter=3000).fit(flat(X_train), y_train)
    per_seed["Logistic regression"].append(accuracy_table(lambda A: lr.predict(flat(A)), X_test, y_test))

    for name, augment in (("CNN", None), ("CNN + shift augmentation", augment_batch)):
        for seed in range(N_SEEDS):
            model = DigitCNN(n_filters=N_FILTERS, seed=seed)
            val_acc, epoch = model.fit(X_train, y_train, X_val, y_val, augment=augment, epochs=100, seed=seed)
            table = accuracy_table(model.predict, X_test, y_test)
            per_seed[name].append(table)
            print(f"{name:26s} seed {seed}: val {val_acc:.3f} (epoch {epoch}) | " +
                  " | ".join(f"{k} {v:.3f}" for k, v in table.items()))
            if seed == 0:
                path = MODELS_DIR / ("cnn_augmented.npz" if augment else "cnn_standard.npz")
                model.save(path)
                shipped[name] = table
                cm = np.zeros((10, 10), dtype=int)
                for t, p in zip(y_test, model.predict(X_test)):
                    cm[t, p] += 1
                confusion[name] = cm.tolist()

    summary = {name: {cond: {"mean": float(np.mean([r[cond] for r in runs])),
                             "std": float(np.std([r[cond] for r in runs]))}
                      for cond in CONDITIONS}
               for name, runs in per_seed.items()}
    results = {
        "dataset": {"train": len(y_train), "validation": len(y_val), "test": len(y_test), "image_size": 8},
        "n_filters": N_FILTERS, "n_seeds": N_SEEDS,
        "summary": summary, "shipped_seed0": shipped, "confusion": confusion,
    }
    (MODELS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print("\nMean accuracy over seeds:")
    for name, table in summary.items():
        print(f"  {name:26s}", "  ".join(f"{c}: {v['mean']:.3f}" for c, v in table.items()))


if __name__ == "__main__":
    main()
