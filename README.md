# CNN from Scratch on Handwritten Digits Classification Problem

A convolutional neural network **built from scratch in NumPy**, with an interactive app that answers one question: *how does a neural network read a handwritten digit?*

<!-- Live demo (after deploying to Streamlit Community Cloud): [Open the app](https://your-app-url.streamlit.app) -->

## What's in the app

Pick a digit in the sidebar; every tab uses it.

1. **Follow a digit:** the digit goes in as 8×8 numbers; filters (small 3×3 patterns the network learned) slide over it and light up where they match; pooling shrinks the result; a final layer picks the answer.
2. **How a filter works:** step a filter across the digit one position at a time and watch the multiply-and-add that produces each output pixel.
3. **Try to fool it:** shift the digit or add noise and see whether two networks (a standard CNN and one trained with shifted digits) still get it right, plus how they score across all test digits.

Colours mean one thing everywhere: green = present or strong, gold = a filter wants blank there, grey = not computed yet, red = the current filter window.

## Run it

Requires Python 3.9+. Trained models and the test images are included.

```bash
git clone https://github.com/hassanayoub409/cnn-from-scratch.git
cd cnn-from-scratch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Retrain everything (about a minute): `pip install -r requirements-train.txt && python train.py`. Run the tests: `pip install pytest && python -m pytest -q`.

## What's interesting about the implementation

- **Everything from scratch in NumPy:** vectorized convolution (with stride and padding support), max pooling, dense layer, softmax cross-entropy, backpropagation through all of them, and the Adam optimizer (`src/`).
- **Verified against the textbook definition:** the vectorized convolution is tested against the four-nested-loops version for several stride/padding settings, and all gradients (convolution, pooling, full model) are checked against numerical gradients in `tests/`.
- **Honest evaluation:** several seeds per variant, a held-out test set never used for early stopping, and a non-CNN baseline to compare against.
- **Visualizations without matplotlib:** every image in the app is rendered from NumPy arrays (`src/viz.py`).

## Project structure

```
app.py              # Streamlit app
train.py            # Trains 2 CNN variants x 5 seeds + baseline, evaluates robustness
src/layers.py       # Conv, pooling and dense layers with backprop
src/model.py        # The CNN: CONV -> ReLU -> MAXPOOL -> DENSE -> softmax
src/transforms.py   # Shift, noise and augmentation
src/viz.py          # Array -> colour image helpers
src/optim.py        # Adam
models/             # Trained weights + results.json
data/               # The 360 held-out test images (for the app)
tests/              # Unit and gradient tests
```

## Data

The 8×8 handwritten digits from the UCI *Optical Recognition of Handwritten Digits* dataset (1,797 images; bundled with scikit-learn). E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, licensed CC BY 4.0. The repository includes only the held-out test images used by the app.

## Limitations

The images are tiny and centered, so this is a teaching model, not a handwriting reader. The shift and noise tests are synthetic.

## Acknowledgments

The concepts behind this project (convolution, pooling and their backpropagation) were learned in the *Convolutional Neural Networks* course of the Deep Learning Specialization by DeepLearning.AI on Coursera. The code in this repository was written independently and contains no course materials or course datasets. This project is not affiliated with or endorsed by Coursera or DeepLearning.AI.

## License

Code: MIT, see [LICENSE](LICENSE). The data is covered by its own license (CC BY 4.0).