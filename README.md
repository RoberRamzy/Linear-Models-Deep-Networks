# Linear-Models-Deep-Networks

A collection of experiments and example implementations for linear classifiers and simple neural networks on MNIST. The notebook `classifier.ipynb` contains sections for:

- Part A — Linear models (logistic regression, softmax regression)
- Part B — Feed-forward neural networks (training loop, per-epoch statistics and plotting)
- Part C — Analysis & hyperparameter sweeps (learning rate, batch size, architecture)
- Part D — Advanced techniques (dropout, CNNs, batch-norm comparisons)

This repository is intended for experimentation, teaching, and quick prototyping.

---

## Quick setup (Linux)

This project uses a Python virtual environment. The repository contains a `.venv` used during development, but you can create your own local venv if you prefer.

Recommended steps (run from repo root):

```bash
# 1) Create a virtual environment (optional if .venv already exists)
python -m venv .venv

# 2) Activate the venv
source .venv/bin/activate

# 3) Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# 4) Install common dependencies
python -m pip install numpy pandas matplotlib scikit-learn jupyter notebook

# 5) Install PyTorch & torchvision
#    For CUDA / platform specific builds follow the official instructions at https://pytorch.org/get-started/locally/
#    Example CPU-only install (change to appropriate command for your system):
python -m pip install torch torchvision
```

Notes:
- PyTorch releases and package names vary by OS and CUDA version. For best results, follow the command generator on https://pytorch.org.
- If you need `scipy` (used for optional stats), install `scipy` as well: `pip install scipy`.

---

## Running the notebook

1. Activate the venv (see above).
2. Launch Jupyter and open `classifier.ipynb`:

```bash
jupyter notebook classifier.ipynb
```

3. Run the notebook cells in order. Key sections to run for plotting and analysis:
   - Part B: Neural Network Implementation — defines `SimpleFNN` and the Part B training loop.
   - The updated Part B training loop now records per-batch losses and accuracies and computes per-epoch mean and std (so you can plot error bands).
   - The plotting cell in Part B draws:
     - Training & validation loss over epochs
     - Training & validation accuracy over epochs
     - Learning curves with error bars (per-epoch std across batches — computed during training)
     - Convergence analysis (linear slope of the last N epochs printed to the output)

Important: run the training cells before the plotting cell so the history lists (e.g., `train_loss_history`, `val_loss_history`, and their `_std_history`) are available.

---

## What the plots show and how they are computed

- Training/Validation loss and accuracy: epoch-wise means computed from per-batch values.
- Error bands / errorbars: per-epoch standard deviation computed across batches within that epoch. This gives intra-epoch variability (useful proxy). For true inter-run uncertainty, run the same experiment multiple times and compute mean ± std across runs.
- Convergence analysis: a simple linear fit (slope) over the last N epochs for the loss curves is printed — negative slope indicates decreasing loss; magnitude indicates speed of decrease.

If you prefer weighted means or variance weighted by batch size, or per-epoch statistics computed across repeated runs, I can add that.

---

## Reproducibility / tips

- To get errorbars from repeated experiments, run the training multiple times (e.g., 3–5 seeds) and average the epoch-wise metrics across runs.
- Use smaller `NUM_EPOCHS` and fewer samples for quick smoke tests.
- If using a GPU, ensure your PyTorch install matches your CUDA version and that the device is available (the notebook uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Files of interest

- `classifier.ipynb` — main notebook with data loading, training loops, plotting, analyses.
- `data/MNIST/raw/` — raw MNIST files (if you prefer to supply dataset manually).

