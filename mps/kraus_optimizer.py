import torch
from torch import nn
import numpy as np


def project_riemannian(Ki, Gi, eta):
    # 2) Optional: normalize or scale G'
    g_norm = torch.linalg.norm(Gi)
    Gi = Gi / g_norm


    # 3) Form the stacked A and B
    #    K, G are each (n, m). We'll stack horizontally => shape (n, 2m).
    A = torch.cat([Gi, Ki], dim=1)  # [n, 2m]
    B = torch.cat([Ki, -Gi], dim=1) # [n, 2m]

    # 4) Build some helper products
    #    B^dagger A => shape (2m, 2m)
    B_dag = B.conj().T  # shape (2m, n)


    # 5) Evaluate the retracted gradient
    #    ∇*_K L(K) = A ( I + (eta/2)*B^dagger A )^-1 B^dagger K
    m = Ki.shape[0]
    I_2m = torch.eye(2*m, dtype=Ki.dtype, device=Ki.device)
    return A @ torch.linalg.inv(I_2m + 0.5*eta*B_dag @ A) @ B_dag @ Ki
    # # shape (2m, 2m)

    # # B^dagger K => shape (2m, m)
    # BdagK = B_dag @ Ki

    # # So new_grad => shape (n, m)
    # new_grad = A @ (M_inv @ BdagK)

    # return new_grad

def retraction(Ki, step):
    """
    Retract Ki + step back onto {M | M^dagger M = I}.
    E.g. simple QR-based retraction for a square matrix.
    """
    M = Ki - step
    return M


class Adam(nn.Module):
    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        params: e.g. a nn.ParameterList of shape (L,) 
                each of shape (K, d^n, d^n).
        """
        super().__init__()
        # We'll store a reference to all param Tensors
        # (like the standard PyTorch optim API does).
        self.params = list(params)  
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # Build Adam buffers (m,v) shaped like each param
        self.m = []
        self.v = []
        for p in self.params:
            # p shape => (K, d^n, d^n)
            self.m.append(torch.zeros_like(p))
            self.v.append(torch.zeros_like(p))

    def step(self):
        self.t += 1
        with torch.no_grad():
            for idx, p in enumerate(self.params):
                if p.grad is None: 
                    continue
                # p => shape (K, d^n, d^n)
                # We'll do a block-wise retraction for each Kraus K_i
                # i.e. for i in [0..K-1]:
                K_blocks = p
                G_blocks = p.grad  # same shape
                new_blocks = []
                for i in range(K_blocks.shape[0]):
                    Ki = K_blocks[i]
                    Gi = G_blocks[i]

                    # Riemannian gradient
                    Griem = project_riemannian(Ki, Gi, self.lr)

                    # # Update Adam buffers
                    # self.m[idx][i] = self.beta1*self.m[idx][i] + (1-self.beta1)*Griem
                    # self.v[idx][i] = self.beta2*self.v[idx][i] + (1-self.beta2)*(Griem.conj()*Griem).real

                    # # Bias correction
                    # mh = self.m[idx][i] / (1 - self.beta1**self.t)
                    # vh = self.v[idx][i] / (1 - self.beta2**self.t)

                    # # Adam step
                    # step = self.lr * mh / (torch.sqrt(vh) + self.eps)

                    # Retraction
                    Ki_new = retraction(Ki, Griem * self.lr)
                    new_blocks.append(Ki_new)

                # stack them back
                new_blocks_t = torch.stack(new_blocks, dim=0)
                p.copy_(new_blocks_t)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()



import torch

def cayley_momentum_update(
    X: torch.Tensor,
    M: torch.Tensor,
    G: torch.Tensor,
    beta: float,
    l: float,
    q: float,
    s: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs one Cayley-SGD with Momentum step on a single block X.

    Returns:
      X_new: updated matrix (same shape as X).
      M_new: updated momentum matrix (same shape as M).
    """

    # 1) Update the momentum: M_{k+1} = beta*M_k - G(X_k)
    M_new = beta*M - G

    # 2) Compute the auxiliary matrix W-hat:
    #    W-hat = M_{k+1} X^T - (1/2) X (X^T M_{k+1} X^T)
    #    Then W = W-hat - W-hat^T   (making it skew-Hermitian)
    #    Finally M_{k+1} = W X_k
    # (We assume real or complex; for complex, "transpose" below might need conj().T if truly unitary.)
    X_t = X.transpose(-2, -1).conj()
    W_hat = M_new @ X_t - 0.5 * X @ (X_t @ M_new @ X_t)
    W = W_hat - W_hat.transpose(-2, -1).conj()

    M_new = W @ X

    # 3) Compute step size alpha = min{ l, 2q/(||W|| + eps) }
    #    We'll use the Frobenius norm (or any consistent norm) for ||W||.
    W_norm = W.norm(p="fro")
    alpha = min(l, 2.0*q/(W_norm.item() + eps))

    # 4) Iterative estimation of the Cayley transform:
    #    Y^0 = X + alpha*M_{k+1}
    #    for i in 1..s:
    #       Y^i = X + (alpha/2) * W (X + Y^{i-1})
    #    X_{k+1} = Y^s
    # (This is a truncated series for (I - alpha/2 W)^{-1} (I + alpha/2 W).)
    Y = X + alpha*M_new  # Y^0
    for i in range(s):
        Y = X + 0.5*alpha * W @ (X + Y)

    X_new = Y

    return X_new, M_new


class CayleySGDMomentum(nn.Module):
    """
    Cayley SGD with Momentum for a list of Stiefel-manifold parameters.
    Each parameter p is a single matrix X in Stiefel (e.g. X^T X=I).
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        beta=0.9,
        q=0.5,
        s=2,
        eps=1e-8,
    ):
        """
        Args:
          params (iterable): List of Tensors, each on the Stiefel manifold
                             (e.g. shape (d,d), or (n,m) with X^T X=I).
          lr (float): Base learning rate 'l' in the algorithm.
          beta (float): Momentum coefficient.
          q (float): Factor for the adaptive step alpha = min{ lr, 2q/(||W||+eps) }.
          s (int): Number of inner iterations for approximating Cayley transform.
          eps (float): Small offset for numerical stability.
        """
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.q = q
        self.s = s
        self.eps = eps

        # Allocate momentum buffers M for each param
        self.M = []
        for p in self.params:
            # p is one matrix on Stiefel => shape e.g. (d, d)
            self.M.append(torch.zeros_like(p))

    def step(self):
        # We disable gradient tracking for the update
        with torch.no_grad():
            for idx, p in enumerate(self.params):
                if p.grad is None:
                    continue
                # Current matrix X, momentum M, Euclidean gradient G
                X = p
                M_ = self.M[idx]
                G = p.grad

                # Cayley + momentum
                X_new, M_new = cayley_momentum_update(
                    X, M_, G,
                    beta=self.beta,
                    l=self.lr,
                    q=self.q,
                    s=self.s,
                    eps=self.eps,
                )

                # Update in-place
                p.copy_(X_new)
                self.M[idx].copy_(M_new)


    def zero_grad(self):
        """
        Zeros all gradients of the parameters.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    

def cayley_adam_update(
    X: torch.Tensor,
    M: torch.Tensor,
    v: torch.Tensor,
    G: torch.Tensor,
    beta1: float,
    beta2: float,
    l: float,
    q: float,
    s: int,
    step: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs one Cayley-Adam step on a single Stiefel parameter block X,
    following Algorithm 2 in the screenshot.

    Args:
      X: current Stiefel-matrix parameter, shape (..., d, d) or (d, d).
      M: first-moment buffer (same shape as G, though we store as shape of X).
      v: second raw moment buffer (scalar, if following the pseudo-code).
      G: Euclidean gradient of X.
      beta1, beta2: Adam momentum coefficients.
      l: base learning rate.
      q: factor for alpha = min{ l, 2q / (||W|| + eps) }.
      s: number of inner iterations for the truncated Cayley transform.
      step: current iteration count (used for bias correction).
      eps: small constant for numerical stability.

    Returns:
      X_new: updated Stiefel-matrix (same shape as X).
      M_new: updated first-moment buffer (same shape as M).
      v_new: updated second raw moment (scalar).
    """

    # ----------------------------------------------------------
    # (4) M_{k+1} <- beta1 * M_k + (1 - beta1) * G(X_k)
    # ----------------------------------------------------------
    M_new = beta1 * M + (1.0 - beta1) * G

    # ----------------------------------------------------------
    # (5) v_{k+1} <- beta2 * v_k + (1 - beta2) * ||G(X_k)||^2
    #     (Here, we store v as a scalar if following the paper.)
    # ----------------------------------------------------------
    g_norm_sq = G.norm(p="fro")**2
    v_new = beta2 * v + (1.0 - beta2) * g_norm_sq

    # ----------------------------------------------------------
    # (6) v_hat_{k+1} = v_{k+1} / (1 - beta2^{k+1})
    # (7) r <- (1 - beta1^{k+1}) * sqrt(v_hat_{k+1}) + eps
    # ----------------------------------------------------------
    # Note: in code, 'step' is the 1-based iteration index.
    # So we use (step) in place of (k+1).
    v_hat = v_new / (1.0 - beta2**step)
    r = (1.0 - beta1**step) * torch.sqrt(v_hat + eps)

    # ----------------------------------------------------------
    # (8) W_hat = M_{k+1} X_k^T - (1/2) X_k (X_k^T M_{k+1} X_k^T)
    # (9) W_k = ( W_hat - W_hat^T ) / r
    #     (skew-symmetric projection)
    # ----------------------------------------------------------
    X_t = X.transpose(-2, -1).conj()   # conj-transpose if complex
    W_hat = M_new @ X_t - 0.5 * X @ (X_t @ M_new @ X_t)
    W = W_hat - W_hat.transpose(-2, -1).conj()
    W = W / r

    # ----------------------------------------------------------
    # (10) M_{k+1} = r * W_k X_k   (store M_{k+1} again with skew-sym)
    # ----------------------------------------------------------
    M_new = r * (W @ X)

    # ----------------------------------------------------------
    # (11, 12) alpha = min{ l, 2q / (||W|| + eps) }
    # ----------------------------------------------------------
    W_norm = W.norm(p="fro")
    alpha = min(l, 2.0 * q / (W_norm.item() + eps))

    # ----------------------------------------------------------
    # (13) Y^0 = X_k - alpha * M_{k+1}
    # (14) for i in 1..s: Y^i = X_k - (alpha/2) * W_k ( X_k + Y^{i-1} )
    # (15) X_{k+1} = Y^s
    # ----------------------------------------------------------
    Y = X - alpha * M_new  # Y^0
    for _ in range(s):
        Y = X - 0.5 * alpha * W @ (X + Y)

    X_new = Y
    return X_new, M_new, v_new

class CayleyAdam(nn.Module):
    """
    Cayley-ADAM for a list of Stiefel-manifold parameters, following Algorithm 2.
    Each parameter p is a matrix X with p^T p = I (real-orthonormal or complex-unitary).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        q=0.5,
        s=2,
        eps=1e-8,
    ):
        """
        Args:
          params (iterable): List of Tensors (each on the Stiefel manifold).
          lr (float): Base learning rate 'l' in the pseudocode.
          betas (tuple): (beta1, beta2) for Adam momentum coefficients.
          q (float): Factor for alpha = min{ lr, 2q/(||W||+eps) }.
          s (int): Number of inner steps in the Cayley transform approximation.
          eps (float): Numerical stability constant.
        """
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.q = q
        self.s = s
        self.eps = eps

        # 't' is the iteration counter (k+1 in the paper).
        self.t = 0

        # Allocate the buffers: M for first moment, v for second raw moment
        self.M = []
        self.v = []
        for p in self.params:
            self.M.append(torch.zeros_like(p))  
            # The algorithm stores v_k as a scalar. We'll just keep it as a 0-D tensor:
            self.v.append(torch.tensor(0.0, dtype=p.dtype, device=p.device))

    def step(self):
        """
        Performs a single ADAM step for each parameter on the Stiefel manifold.
        """
        self.t += 1  # increment global step
        with torch.no_grad():
            for idx, p in enumerate(self.params):
                if p.grad is None:
                    continue

                X = p
                M_ = self.M[idx]
                v_ = self.v[idx]
                G = p.grad

                # 1 step of Cayley-Adam
                X_new, M_new, v_new = cayley_adam_update(
                    X=X,
                    M=M_,
                    v=v_,
                    G=G,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    l=self.lr,
                    q=self.q,
                    s=self.s,
                    step=self.t,
                    eps=self.eps,
                )

                # In-place update
                p.copy_(X_new)
                self.M[idx].copy_(M_new)
                self.v[idx].copy_(v_new)

    def zero_grad(self):
        """
        Zeros out all gradient buffers.
        """
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()