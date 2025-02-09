import torch
import geoopt
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Problem setup
dim_n = 10  # Rows
dim_p = 10   # Columns (must be <= dim_n)
A = torch.randn(dim_p, dim_n)  # Random matrix A of shape (dim_p, dim_n)
B = torch.randn(dim_n, dim_n)  # Target matrix B

# Define the Stiefel manifold
stiefel_manifold = geoopt.manifolds.Stiefel()

# Initialize a point on the Stiefel manifold
X_init_o = torch.randn(dim_n, dim_p)
X_init, _ = torch.qr(X_init_o)  # Orthogonalize to be on the manifold

# Create an initial Euclidean parameter
w_init = torch.randn(dim_n, dim_n)

# Define the modified loss function that depends on both the manifold variable X and the Euclidean parameter w.
def quadratic_loss(X, w):
    # The loss measures how well |X @ A + w| approximates B.
    return torch.norm(torch.abs(X @ A) - B, p='fro') ** 2 + torch.norm(w, p='fro') ** 2

# Training loop settings
max_iters = 5000
convergence_threshold = 1e-6
patience = 30

# A custom EarlyStopping class (a strategy often recommended in PyTorch code)
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-6):
        """
        Args:
            patience (int): How many iterations to wait after last improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        # If this is the first call, record the best loss.
        if self.best_loss is None:
            self.best_loss = current_loss
        # If the loss improves by more than delta, reset the counter.
        elif current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            # If no improvement for 'patience' consecutive iterations, signal to stop.
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop function for a given optimizer and parameters X and w.
# It returns the loss history and the gradient norm history of w.
def train(optimizer, X, w, name):
    loss_history = []
    grad_history = []
    w_first_element_history = []
    
    # Create an EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, delta=convergence_threshold)
    
    for i in range(max_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(X, w)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        grad_history.append(w.grad.norm().item())
        w_first_element_history.append(w[0, 0].item())
        
        # Check early stopping condition
        early_stopping(loss_val)
        if early_stopping.early_stop:
            print(f"{name} converged at iteration {i+1} with loss {loss_val:.6f}")
            break

    return loss_history, grad_history, w_first_element_history

# Dictionaries to store loss and gradient histories for each optimizer configuration.
loss_histories = {}
grad_histories = {}
w_first_element_histories = {}
lr = 0.01  # Learning rate

# Define beta combinations to try for Riemannian Adam.
beta_combinations = {
    "Default (0.9, 0.999)": (0.9, 0.999),
    "Beta (0.9, 0.9)"      : (0.9, 0.9),
    "Beta (0.5, 0.999)"    : (0.5, 0.999),
    "Beta (0.5, 0.5)"      : (0.5, 0.5),
    "Beta (0.8, 0.9)"      : (0.8, 0.9),
    "Beta (0.7, 0.8)"      : (0.7, 0.8),
    "Beta (0.6, 0.7)"      : (0.6, 0.7),
    "Beta (0.4, 0.6)"      : (0.4, 0.6)
}

# Loop over each beta setting and run training for Riemannian Adam.
for beta_name, beta_vals in beta_combinations.items():
    # Each run starts from the same initial point (cloned) to ensure fairness.
    X_temp = geoopt.ManifoldParameter(X_init.clone(), manifold=stiefel_manifold)
    w_temp = torch.nn.Parameter(w_init.clone())
    adam_optimizer = geoopt.optim.RiemannianAdam([X_temp, w_temp], lr=lr, betas=beta_vals)
    print(f"Running training with {beta_name} (betas={beta_vals})")
    loss_history, grad_history, w_first_element_history = train(adam_optimizer, X_temp, w_temp, f"Riemannian Adam {beta_name}")
    loss_histories[f"Riemannian Adam {beta_name}"] = loss_history
    grad_histories[f"Riemannian Adam {beta_name}"] = grad_history
    w_first_element_histories[f"Riemannian Adam {beta_name}"] = w_first_element_history

# Run training with Riemannian SGD as a baseline (using both parameters).
X_temp_sgd = geoopt.ManifoldParameter(X_init.clone(), manifold=stiefel_manifold)
w_temp_sgd = torch.nn.Parameter(w_init.clone())
sgd_optimizer = geoopt.optim.RiemannianSGD([X_temp_sgd, w_temp_sgd], lr=lr)
print("Running training with Riemannian SGD")
loss_history_sgd, grad_history_sgd, w_first_element_history_sgd = train(sgd_optimizer, X_temp_sgd, w_temp_sgd, "Riemannian SGD")
loss_histories["Riemannian SGD"] = loss_history_sgd
grad_histories["Riemannian SGD"] = grad_history_sgd
w_first_element_histories["Riemannian SGD"] = w_first_element_history_sgd

# Plot the loss histories for comparison.
plt.figure(figsize=(10, 6))
for name, loss_history in loss_histories.items():
    plt.plot(loss_history, label=name, alpha=0.8)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Comparison of Riemannian Optimizers on the Stiefel Manifold (Mixed Parameters)')
plt.legend()
plt.show()

# Plot the gradient norm histories of w to see how they converge to zero.
plt.figure(figsize=(10, 6))
for name, grad_history in grad_histories.items():
    plt.plot(grad_history, label=name, alpha=0.8)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Gradient norm of w (log scale)')
plt.title('Convergence of the Gradient Norm for the Euclidean Parameter w')
plt.legend()
plt.show()

# Plot the first element of w over iterations.
plt.figure(figsize=(10, 6))
for name, w_first_element_history in w_first_element_histories.items():
    plt.plot(w_first_element_history, label=name, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('First element of w')
plt.title('Change of the First Element of w Over Iterations')
plt.legend()
plt.show()
