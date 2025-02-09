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
    return torch.norm(torch.abs(X @ A + w) - B @ w, p='fro') ** 2

# Training loop settings
max_iters = 5000
convergence_threshold = 1e-6
patience = 10

# Training loop function for a given optimizer and parameters X and w.
# It returns both the loss history and the gradient norm history of w.
def train(optimizer, X, w, name):
    loss_history = []
    grad_history = []
    w_first_element_history = []
    best_loss = float('inf')
    no_improve_count = 0
    for i in range(max_iters):
        optimizer.zero_grad()
        loss = quadratic_loss(X, w)
        loss.backward()
        # Record the norm of the gradient of the Euclidean parameter w.
        print(f"w.grad: {w.grad}")
        print(f"X.grad: {X.grad}")
        optimizer.step()
        loss_history.append(loss.item())
        grad_history.append(w.grad.norm().item())
        w_first_element_history.append(w[0, 0].item())
        
        # Check convergence criteria.
        # if loss.item() < best_loss - convergence_threshold:
        #     best_loss = loss.item()
        #     no_improve_count = 0
        # else:
        #     no_improve_count += 1
        
        # if no_improve_count >= patience:
        #     print(f"{name} converged at iteration {i+1} with loss {loss.item():.6f}")
        #     break
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
