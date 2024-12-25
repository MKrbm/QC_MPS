import torch
import numpy as np


def riemannian_gradient(u: torch.Tensor, euc_grad: torch.Tensor) -> torch.Tensor:
    """
    Requires square matrices for u and euc_grad.
    """
    grad = euc_grad @ u.T.conj()
    return (grad - grad.T.conj()) / 2

def exp_map(u: torch.Tensor, rg: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on the unitary manifold.
    """
    res =  torch.linalg.matrix_exp(-rg) @ u
    return closest_unitary(res)

def closest_unitary(u: torch.Tensor) -> torch.Tensor:
    """
    Find the closest unitary to u.
    """
    U, S, Vh = torch.linalg.svd(u)
    return U @ Vh


class SGD:
    """
    A simple optimizer for training the unitaries of a CircuitMPS instance using gradient descent.
    Ensures that the unitaries remain unitary after each update.
    """
    
    def __init__(self, circuit, lr: float = 0.01):
        """
        Initializes the optimizer.

        Args:
            circuit (CircuitMPS): The CircuitMPS instance to optimize.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        """
        self.circuit = circuit
        self.lr = lr
        # Ensure that the MPS unitaries have gradients enabled
        for param in self.circuit.parameters():
            if not param.requires_grad:
                param.requires_grad = True

    def step(self):
        """
        Performs a single optimization step (parameter update).
        Applies gradient descent to each unitary and re-unitarizes them.
        """
        chi = self.circuit.chi
        with torch.no_grad():
            for param in self.circuit.parameters():
                if param.grad is not None:
                    # Gradient descent update
                    u = param.view(chi**2, chi**2)
                    euc_grad = param.grad.view(chi**2, chi**2)
                    rg = riemannian_gradient(u, euc_grad)
                    rg = self.lr * rg
                    u_updated = exp_map(u, rg)
                    param.copy_(u_updated.view(param.shape))

    def zero_grad(self):
        """
        Zeros the gradients of all MPS unitaries.
        """
        for param in self.circuit.params:
            if param.grad is not None:
                param.grad.zero_()


class Adam:
    """
    An optimizer for training the unitaries of a CircuitMPS instance using Adam,
    adapted for the unitary group to ensure updates remain on the manifold.
    """

    def __init__(self, circuit, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the optimizer.

        Args:
            circuit (CircuitMPS): The CircuitMPS instance to optimize.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            beta1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
            epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
        """
        self.circuit = circuit
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step

        # Initialize first and second moment estimates
        self.m = []
        self.v = []
        # Ensure that the MPS unitaries have gradients enabled and initialize m and v
        for unitary in self.circuit.params:
            if not unitary.requires_grad:
                unitary.requires_grad = True
            self.m.append(torch.zeros_like(unitary))
            self.v.append(torch.zeros_like(unitary))

    def step(self):
        """
        Performs a single optimization step (parameter update).
        Applies Adam optimizer to each unitary, ensuring that the updates stay on the unitary manifold.
        """
        chi = self.circuit.chi
        self.t += 1  # Increment time step
        with torch.no_grad():
            for idx, unitary in enumerate(self.circuit.params):
                if unitary.grad is not None:
                    # Reshape unitary and gradient
                    u = unitary.view(chi**2, chi**2)
                    euc_grad = unitary.grad.view(chi**2, chi**2)

                    # Compute the Riemannian gradient (skew-Hermitian part)
                    gt = riemannian_gradient(u, euc_grad)

                    # Update biased first moment estimate
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * gt.view(unitary.shape)

                    # Update biased second raw moment estimate (element-wise square of gradient)
                    gt_squared = gt.abs() ** 2
                    self.v[idx] = (
                        self.beta2 * self.v[idx] + (1 - self.beta2) * gt_squared.view(unitary.shape)
                    )

                    # Compute bias-corrected first moment estimate
                    mhat = self.m[idx] / (1 - self.beta1**self.t)

                    # Compute bias-corrected second raw moment estimate
                    vhat = self.v[idx] / (1 - self.beta2**self.t)

                    # Compute the update
                    rg = self.lr * mhat / (torch.sqrt(vhat) + self.epsilon)

                    # Apply the exponential map to update the unitary
                    u_updated = exp_map(u, rg.view(chi**2, chi**2))

                    # Update the unitary in the circuit
                    unitary.copy_(u_updated.view(unitary.shape))

    def zero_grad(self):
        """
        Zeros the gradients of all MPS unitaries.
        """
        for unitary in self.circuit.params:
            if unitary.grad is not None:
                unitary.grad.zero_()



class AdamUnitary:
    """
    An optimizer for training the unitaries of a CircuitMPS instance using Adam,
    adapted for the unitary group to ensure updates remain on the manifold.
    """

    def __init__(self, params : list[torch.Tensor], lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the optimizer.

        Args:
            params (list[torch.Tensor]): The parameters to optimize.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            beta1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
            epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step

        assert all(param.dim() == 4 for param in self.params), "All parameters must be tensor of rank 4"
        self.chi = self.params[0].shape[0]
        assert all(param.shape == (self.chi, self.chi, self.chi, self.chi) for param in self.params), "All parameters must have the same chi"

        # Initialize first and second moment estimates
        self.m = []
        self.v = []
        # Ensure that the MPS unitaries have gradients enabled and initialize m and v
        for param in self.params:
            if not param.requires_grad:
                param.requires_grad = True
            # ensure param is a square matrix
            self.m.append(torch.zeros_like(param))
            self.v.append(torch.zeros_like(param))

    def step(self):
        """
        Performs a single optimization step (parameter update).
        Applies Adam optimizer to each unitary, ensuring that the updates stay on the unitary manifold.
        """
        chi = self.chi
        self.t += 1  # Increment time step
        with torch.no_grad():
            for idx, unitary in enumerate(self.params):
                if unitary.grad is not None:
                    # Reshape unitary and gradient
                    u = unitary.view(chi**2, chi**2)
                    euc_grad = unitary.grad.view(chi**2, chi**2)

                    # Compute the Riemannian gradient (skew-Hermitian part)
                    gt = riemannian_gradient(u, euc_grad)

                    # Update biased first moment estimate
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * gt.view(unitary.shape)

                    # Update biased second raw moment estimate (element-wise square of gradient)
                    gt_squared = gt.abs() ** 2
                    self.v[idx] = (
                        self.beta2 * self.v[idx] + (1 - self.beta2) * gt_squared.view(unitary.shape)
                    )

                    # Compute bias-corrected first moment estimate
                    mhat = self.m[idx] / (1 - self.beta1**self.t)

                    # Compute bias-corrected second raw moment estimate
                    vhat = self.v[idx] / (1 - self.beta2**self.t)

                    # Compute the update
                    rg = self.lr * mhat / (torch.sqrt(vhat) + self.epsilon)

                    # Apply the exponential map to update the unitary
                    u_updated = exp_map(u, rg.view(chi**2, chi**2))

                    # Update the unitary in the circuit
                    unitary.copy_(u_updated.view(unitary.shape))

    def zero_grad(self):
        """
        Zeros the gradients of all MPS unitaries.
        """
        for unitary in self.params:
            if unitary.grad is not None:
                unitary.grad.zero_()


