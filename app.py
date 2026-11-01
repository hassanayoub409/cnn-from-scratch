"""Streamlit app: CNN X-Ray. Follow one handwritten digit through a convolutional neural network
that was built from scratch in NumPy.

Run with:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src import viz
from src.layers import conv_forward
from src.model import DigitCNN
from src.paths import DATA_DIR, MODELS_DIR
from src.transforms import add_noise, shift

st.set_page_config(page_title="CNN X-Ray", page_icon="🔍", layout="wide")

STANDARD, TRAINED = "Standard CNN", "CNN trained with shifted digits"
RESULT_NAMES = {"Logistic regression": "Baseline (no convolution)", "CNN": STANDARD,
                "CNN + shift augmentation": TRAINED}
CONDITION_NAMES = {"clean": "Clean digits", "shift 1px": "Shifted 1 pixel", "shift 2px": "Shifted 2 pixels",
                   "noise 0.1": "Noise: low", "noise 0.2": "Noise: medium", "noise 0.3": "Noise: high"}


@st.cache_resource
def load_everything():
    models = {STANDARD: DigitCNN.load(MODELS_DIR / "cnn_standard.npz"),
              TRAINED: DigitCNN.load(MODELS_DIR / "cnn_augmented.npz")}
    test = np.load(DATA_DIR / "digits_test.npz")
    results = json.loads((MODELS_DIR / "results.json").read_text())
    return models, test["X"].astype(float) / 16.0, test["y"], results


if not (MODELS_DIR / "results.json").exists():
    st.error("No trained models found. Run `python train.py` first (see the README).")
    st.stop()
models, X_test, y_test, results = load_everything()
model = models[STANDARD]


def show(rgb, px):
    """Show a small array as a crisp image about `px` pixels wide."""
    k = max(1, px // rgb.shape[1])
    st.image(viz.upscale(rgb, k), width=k * rgb.shape[1])


# ------------------------------------------------------------------ sidebar: pick a digit
st.sidebar.header("Pick a digit")
st.session_state.setdefault("idx", 0)


def pick_random():
    st.session_state["idx"] = int(np.random.default_rng().integers(len(X_test)))


st.sidebar.button("🎲 Random digit", on_click=pick_random)
idx = st.sidebar.slider("Digit number", 0, len(X_test) - 1, key="idx")
digit = X_test[idx]
true_label = int(y_test[idx])
st.sidebar.image(viz.upscale(viz.intensity(digit), 14), width=14 * 8, caption="Selected digit")
st.sidebar.caption("Every tab uses this digit.")

st.title("CNN from Scratch on a Handwritten Digit Classification Problem")
st.markdown("**How does a neural network read a handwritten digit?** Follow one digit through the network, "
            "see how a filter works, then try to fool it.")

tab_follow, tab_filter, tab_fool, tab_about = st.tabs(
    ["1 · Follow a digit", "2 · How a filter works", "3 · Try to fool it", "About"])

# ------------------------------------------------------------------ 1 · Follow a digit
with tab_follow:
    image = digit[None, :, :, None]
    probs = model.predict_proba(image)[0]
    pred = int(probs.argmax())
    _, acts = model.forward(image)

    st.subheader("Step 1 · The input")
    c1, c2 = st.columns([1, 3])
    with c1:
        show(viz.intensity(digit), 200)
    with c2:
        st.markdown("The network sees an image as an **8×8 grid of numbers**, one brightness value per pixel "
                    "(0 = blank paper, 16 = full ink). That is all it gets.")
        with st.expander("Show the 64 numbers"):
            st.dataframe(pd.DataFrame(np.rint(digit * 16).astype(int)), hide_index=True)

    st.subheader("Step 2 · Filters look for small patterns")
    st.markdown(
        "A **filter** is a tiny 3×3 pattern. The network slides it over the whole image and checks how well each spot "
        "matches the pattern. The result is a map of *where the pattern was found*. The network learned its "
        f"{model.n_filters} filters by itself during training."
    )
    maps = acts["relu"][0]      # (6, 6, filters): where each filter matched
    pooled = acts["pool"][0]    # (3, 3, filters): maps shrunk, keeping the strongest matches
    W = model.conv.W[:, :, 0, :]
    vmax_filter, vmax_map = float(np.abs(W).max()), float(maps.max()) or 1.0
    show_all = st.checkbox(f"Show all {model.n_filters} filters (showing the first 3)")
    n_show = model.n_filters if show_all else 3

    heads = st.columns([0.6, 1, 1, 1])
    heads[1].markdown("**Filter pattern**  \n<small>green = wants ink here · gold = wants blank here</small>",
                      unsafe_allow_html=True)
    heads[2].markdown("**Where it matches this digit**  \n<small>darker green = stronger match</small>",
                      unsafe_allow_html=True)
    heads[3].markdown("**After pooling**  \n<small>smaller map, strongest matches kept</small>",
                      unsafe_allow_html=True)
    for k in range(n_show):
        row = st.columns([0.6, 1, 1, 1])
        row[0].markdown(f"<br><b>Filter {k + 1}</b>", unsafe_allow_html=True)
        with row[1]:
            show(viz.filter_image(W[:, :, k], vmax_filter), 90)
        with row[2]:
            show(viz.intensity(maps[:, :, k], vmax_map), 96)
        with row[3]:
            show(viz.intensity(pooled[:, :, k], vmax_map), 90)

    st.subheader("Step 3 · The decision")
    st.markdown(
        f"All {model.n_filters} pooled maps ({pooled.size} numbers in total) go into a final layer that gives every "
        "digit from 0 to 9 a score. **The highest score wins.**"
    )
    d1, d2 = st.columns([1, 3])
    d1.metric("The network's answer", f"{pred}", f"{probs[pred]:.0%} sure", delta_color="off")
    with d2:
        if pred == true_label:
            st.success(f"Correct: this digit really is a {true_label}.")
        else:
            st.error(f"Wrong: this digit is actually a {true_label}.")

# ------------------------------------------------------------------ 2 · How a filter works
FILTERS = {
    "Vertical lines": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
    "Horizontal lines": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
    "Smooth (blur)": [[1 / 9] * 3] * 3,
}

with tab_filter:
    st.markdown(
        "A filter slides across the image. At each position it **multiplies** the 9 pixels under it by the filter's "
        "9 numbers and **adds them up**. That one sum becomes one pixel of the output map. "
        "These three filters are made by hand to show the idea (the network learns its own)."
    )
    left, right = st.columns([1, 3])
    name = left.selectbox("Filter", list(FILTERS))
    kernel = np.array(FILTERS[name], dtype=float)
    out = conv_forward(digit[None, :, :, None], kernel[:, :, None, None], np.zeros(1))[0, :, :, 0]  # 6 x 6
    n = out.shape[0]
    pos = left.slider("Move the window", 0, n * n - 1, 0, help="Step through every position of the filter.")
    r, c = divmod(pos, n)
    left.caption(f"Window at row {r + 1}, column {c + 1} of {n}×{n} positions.")
    patch = digit[r:r + 3, c:c + 3] * 16
    products = patch * kernel
    total = float(products.sum())

    with right:
        i1, i2, i3 = st.columns(3)
        with i1:
            st.markdown("**The digit**  \n<small>red box = where the filter is</small>", unsafe_allow_html=True)
            st.image(viz.draw_box(viz.upscale(viz.intensity(digit), 28), r, c, 3, 28), width=224)
        with i2:
            st.markdown("**The filter**  \n<small>green = wants ink · gold = wants blank</small>", unsafe_allow_html=True)
            st.image(viz.upscale(viz.filter_image(kernel, 2.0), 56), width=168)
        with i3:
            st.markdown("**The result so far**  \n<small>grey = not computed yet</small>", unsafe_allow_html=True)
            filled = (np.arange(n * n).reshape(n, n) <= pos)
            vmax = max(float(np.maximum(out * 16, 0).max()), 1e-9)
            st.image(viz.draw_box(viz.upscale(viz.partial_map(out * 16, filled, vmax), 36), r, c, 1, 36), width=216)
        t1, t2, t3 = st.columns(3)
        fmt = lambda a: pd.DataFrame(np.round(a, 1))
        t1.markdown("**Pixels under the window**")
        t1.dataframe(fmt(patch), hide_index=True)
        t2.markdown("**× Filter numbers**")
        t2.dataframe(fmt(kernel), hide_index=True)
        t3.markdown("**= Products**")
        t3.dataframe(fmt(products), hide_index=True)
        st.markdown(f"**Add up all 9 products: {total:.1f}.** That number is the output pixel at this position. "
                    "Output pixels that would be negative are shown blank.")

# ------------------------------------------------------------------ 3 · Try to fool it
with tab_fool:
    st.markdown(
        "Real handwriting is never perfectly centered or clean. Move the digit or add noise and see whether the "
        "networks still read it correctly. The second network was trained with slightly shifted digits."
    )
    s1, s2, s3 = st.columns(3)
    dx = s1.slider("Shift right (pixels)", -3, 3, 0)
    dy = s2.slider("Shift down (pixels)", -3, 3, 0)
    sigma = s3.slider("Noise", 0.0, 0.5, 0.0, step=0.05)
    disturbed = add_noise(shift(digit[None, :, :, None], dx, dy), sigma, np.random.default_rng(idx))

    col_a, col_b, col_c = st.columns([1, 1.5, 1.5])
    with col_a:
        st.markdown("**Original**")
        show(viz.intensity(digit), 150)
        st.markdown("**After your changes**")
        show(viz.intensity(disturbed[0, :, :, 0]), 150)
    for col, name in ((col_b, STANDARD), (col_c, TRAINED)):
        p = models[name].predict_proba(disturbed)[0]
        guess = int(p.argmax())
        with col:
            st.markdown(f"**{name}**")
            st.metric("Answer", f"{guess}", f"{p[guess]:.0%} sure", delta_color="off")
            if guess == true_label:
                st.success(f"Correct ({true_label})")
            else:
                st.error(f"Wrong: it is a {true_label}")

    st.divider()
    st.subheader("How often are they right? (all 360 test digits)")
    summary = results["summary"]
    table = pd.DataFrame(
        {RESULT_NAMES[m]: [f"{summary[m][c]['mean']:.0%}" for c in CONDITION_NAMES] for m in summary},
        index=list(CONDITION_NAMES.values()))
    st.dataframe(table)
    base, cnn, aug = summary["Logistic regression"], summary["CNN"], summary["CNN + shift augmentation"]
    st.markdown(
        f"- On clean digits all three are about equally good ({cnn['clean']['mean']:.0%}), so this alone hides the differences.\n"
        f"- **Shifting by just 1 pixel** is enough to break the baseline ({base['shift 1px']['mean']:.0%} right) and hurts the "
        f"standard CNN ({cnn['shift 1px']['mean']:.0%}). Training on shifted digits fixes most of it ({aug['shift 1px']['mean']:.0%}).\n"
        f"- **There is a catch:** the shift-trained network copes *worse* with noise ({aug['noise 0.3']['mean']:.0%} vs "
        f"{cnn['noise 0.3']['mean']:.0%} at high noise). Getting robust to one kind of change does not make it robust to others."
    )
    st.caption(f"CNN numbers are averages over {results['n_seeds']} training runs with different random seeds. "
               "The baseline is a logistic regression that looks at the 64 pixels directly, without filters.")

# ------------------------------------------------------------------ About
with tab_about:
    n_params = {k: v.size for k, v in model.params.items()}
    d = results["dataset"]
    st.markdown(
        f"""
**The network:** `3×3 filters ({model.n_filters}) → keep positive values (ReLU) → pooling (2×2 max) → final scoring layer → 10 digit scores`.
It has {sum(n_params.values()):,} adjustable numbers in total.

**Built from scratch:** filters, pooling, the scoring layer, the learning algorithm (backpropagation and Adam) are all plain
NumPy, in `src/`. The code is tested against the textbook definition of convolution and against numerical gradients.

**Data:** 8×8 handwritten digits from the UCI *Optical Recognition of Handwritten Digits* dataset (1,797 images,
bundled with scikit-learn): {d['train']} for training, {d['validation']} for checking, {d['test']} for testing.

**Limits:** the images are tiny and centered, so this is a teaching model, not a handwriting reader.
The shifts and noise are artificial.
"""
    )
    st.caption(
        "Data: E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning "
        "Repository, CC BY 4.0. The concepts were learned in the Convolutional Neural Networks course (DeepLearning.AI "
        "on Coursera); this code is an independent re-implementation and contains no course materials."
    )
