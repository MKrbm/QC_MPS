# TPCP MPS Training on MNIST

This repository demonstrates training a **Matrix Product State (MPS)**-inspired model, known as **MPSTPCP**, on the **MNIST** dataset (restricted to digits **0** and **1**). The approach leverages **Riemannian optimization** on different **Stiefel manifold** parameterizations (`EXACT`, `FROBENIUS`, `CANONICAL`) via [**geoopt**](https://github.com/geoopt/geoopt).

## Project Structure

```
.
├── mps/
│   ├── __init__.py
│   ├── tpcp_mps.py       # Core implementation of MPSTPCP model
│   ├── unitary_optimizer # (Example submodule for MPS or additional optim code)
│   └── ...               # Additional MPS-related modules
├── tests/
│   └── integrating/
│       ├── test_comp_manifold.py  # Example integration tests
│       └── ...
├── train_tpcp_mnist.py   # Main script to train an MPSTPCP model on MNIST
├── compare_optimizers_tpcp_mnist.py # Example comparing Adam vs SGD on 0/1 MNIST
├── README.md             # This file
├── requirements.txt      # (Optional) Python dependencies
└── ...
```

## Features

- **MPSTPCP Model**  
  - A matrix product operator that enforces **completely positive trace-preserving** constraints site-by-site, using Stiefel manifold-based parameterizations.

- **Riemannian Optimization**  
  - Compatible with [geoopt](https://github.com/geoopt/geoopt) to handle unitary or orthonormal constraints with **`RiemannianSGD`** or **`RiemannianAdam`**.

- **MNIST Demo (Digits 0 & 1)**  
  - Shows how to embed grayscale pixels into a 2D feature space `[x, 1-x]` and train for binary classification.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/tpcp-mps.git
   cd tpcp-mps
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or manually install key libraries:
   ```bash
   pip install torch torchvision geoopt matplotlib
   ```

3. **(Optional) Editable install**:
   ```bash
   pip install -e .
   ```
   This allows Python to discover the `mps` package easily.

---

## Usage

### 1. Basic Training on MNIST

Train the **`MPSTPCP`** model on digits **0** and **1** from MNIST:

```bash
python train_tpcp_mnist.py \
  --manifold EXACT \
  --optimizer adam \
  --K 1 \
  --batch_size 128 \
  --epochs 10 \
  --num_data 20000 \
  --lr 0.005 \
  --seed 42
```

**Available Arguments**:

- `--manifold`: Manifold type, one of **`EXACT`**, **`FROBENIUS`**, **`CANONICAL`**.  
- `--optimizer`: **`adam`** or **`sgd`** (uses geoopt’s Riemannian versions).  
- `--K`: Number of Kraus operators per site.  
- `--batch_size`: Training batch size (default 128).  
- `--epochs`: Number of epochs (default 10).  
- `--num_data`: Limit the dataset size (e.g., 20000). Omit to use all.  
- `--lr`: Learning rate (default 0.01).  
- `--seed`: Random seed (if omitted or `None`, uses time-based randomness).  

A **Matplotlib** plot of the batch-wise training loss vs. iteration will appear after training.

### 2. Optimizer Comparison on a Single Batch

**`compare_optimizers_tpcp_mnist.py`** allows you to visualize how **Adam vs. SGD** performs on a single batch of MNIST, across different manifold parameterizations. For example:

```bash
python compare_optimizers_tpcp_mnist.py
```

It will create subplots showing **loss curves** over a fixed number of steps (`steps=30` by default).

### 3. Tests

We use **pytest** for integration tests:

```bash
pytest tests/
```

Ensure you run this **from the project root**, so Python can locate the `mps` package.

---

## Contributing

1. **Fork** the repository and create your branch from `main`.  
2. **Add** or modify features/tests.  
3. **Open a pull request** describing your changes.

---

## License

(MIT or your chosen license)

---

## References & Acknowledgments

- [geoopt](https://github.com/geoopt/geoopt) for Riemannian optimization.  
- *If relevant*, references to academic papers or prior MPS/QC resources.

---

**Enjoy exploring the TPCP MPS approach for quantum/classical machine learning on MNIST!**