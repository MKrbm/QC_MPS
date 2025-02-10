import torch  # do not change the order of import. otherwise, it causes error at QR factorization.
import opt_einsum as oe
from torch import nn
import numpy as np
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group
import geoopt  # Added for Geoopt functionalities
from enum import Enum

class ManifoldType(Enum):
    CANONICAL = "canonical"
    FROBENIUS = "frobenius"
    EXACT = "exact"


class kraus_operators(nn.Module):

    K: int  # number of kraus operators.
    d: int = 2
    n: int = 2  # number of qubits it acts on.
    L: int  # number of kraus_operator layers.

    def __init__(self, K, L, with_identity: bool = False, init_with: torch.Tensor | None = None, manifold: ManifoldType = ManifoldType.CANONICAL):
        super().__init__()

        if manifold not in ManifoldType:
            raise ValueError(f"Invalid manifold type: {manifold}")

        self.K = K
        self.L = L

        self.act_size = self.d ** self.n
        # self.manifold = geoopt.Stiefel(canonical = (manifold == ManifoldType.CANONICAL))
        if manifold == ManifoldType.EXACT:
            self.manifold = geoopt.EuclideanStiefelExact()
        elif manifold == ManifoldType.FROBENIUS:
            self.manifold = geoopt.EuclideanStiefel()
        elif manifold == ManifoldType.CANONICAL:
            self.manifold = geoopt.CanonicalStiefel()
        else:
            raise ValueError(f"Invalid manifold type: {manifold}")

        # Each parameter is now defined as a Geoopt ManifoldParameter on the Stiefel manifold.
        self.kraus_ops = nn.ParameterList(
            [geoopt.ManifoldParameter(
                torch.zeros(self.K * self.act_size, self.act_size, dtype=torch.float64),
                manifold=self.manifold,
                requires_grad=True
            ) for _ in range(self.L)]
        )
        self.init_params(init_with=init_with, init_with_identity=with_identity)

    def __getitem__(self, idx):
        return self.kraus_ops[idx]

    def init_params(self, init_with: torch.Tensor | None = None, init_with_identity: bool = False):
        """
        Initialize or overwrite the parameters of the Kraus operators.

        Args:
            init_with (torch.Tensor, optional): Tensor to initialize Kraus operators with.
                                                Must represent a point on the Stiefel manifold if provided.
            init_with_identity (bool): If True, initialize each Kraus operator to be proportional
                                       to the identity (normalized by 1/sqrt(K)) so that the TPCP condition holds.
        """
        stiefel = geoopt.Stiefel()
        if init_with is not None:
            if len(init_with) != len(self.kraus_ops):
                print(f"init_with length: {len(init_with)}, expected: {len(self.kraus_ops)}")
                raise ValueError(f"init_with tensor must have length equal to number of layers {len(self.kraus_ops)}")
            for l in range(self.L):
                # Reshape init_with[l] to match the parameter shape: (K * act_size, act_size)
                candidate = init_with[l].reshape(self.K * self.act_size, self.act_size)
                # Use the built-in 'belongs' function to check if candidate is on the Stiefel manifold.
                if not stiefel.check_point_on_manifold(candidate):
                    raise ValueError("The provided tensor does not represent a point on the Stiefel manifold.")
                self.kraus_ops[l].data.copy_(candidate.clone())
        else:
            for l in range(self.L):
                if init_with_identity:
                    # Create identity Kraus operators normalized by 1/sqrt(K)
                    identity = torch.eye(self.act_size, dtype=torch.float64, device=self.kraus_ops[l].device)
                    kraus_identity = (1.0 / np.sqrt(self.K)) * identity
                    param_val = torch.stack([kraus_identity for _ in range(self.K)]).reshape(self.K * self.act_size, self.act_size)
                else:
                    # Use geoopt's Stiefel manifold random generator.
                    param_val = stiefel.random(
                        (self.K * self.act_size, self.act_size),
                        dtype=torch.float64,
                        device=self.kraus_ops[l].device
                    )
                self.kraus_ops[l].data.copy_(param_val)

    @staticmethod
    def is_tpcp(kraus_tensor: torch.Tensor) -> bool:
        """
        Check if the given Kraus tensor represents a TPCP map.

        Args:
            kraus_tensor (torch.Tensor): Tensor containing Kraus operators
                                         with shape (K, d, d).

        Returns:
            bool: True if the tensor is TPCP, False otherwise.
        """
        # Expect kraus_tensor to have shape (K, d, d)
        K, d, d_ = kraus_tensor.shape
        if d != d_:
            return False  # Not a square matrix.
        identity = torch.eye(d, dtype=kraus_tensor.dtype, device=kraus_tensor.device)
        sum_kd = torch.stack([K_i.conj().transpose(-1, -2) @ K_i for K_i in kraus_tensor]).sum(dim=0)
        return torch.allclose(sum_kd, identity, atol=1e-6)

    @staticmethod
    def random_tpcp_map_torch(
        d, k, with_identity=False, dtype=torch.float64, device="cpu"
    ) -> torch.Tensor:
        """
        Generate a random TPCP map in d-dimensional Hilbert space
        with k Kraus operators using the isometry method, in PyTorch.
        (This method is retained for compatibility but is not used in the geoopt initialization.)

        Args:
            d (int): Hilbert space dimension.
            k (int): Number of Kraus operators.
            with_identity (bool): Whether to include the identity operator.
            dtype (torch.dtype): Data type.
            device (str or torch.device): Device for tensors.

        Returns:
            A tensor representing k Kraus operators stacked together with shape (k*d, d).
        """
        if with_identity:
            identity = torch.eye(d, dtype=dtype, device=device)
            Kraus_ops = [identity for _ in range(k)]
            return torch.stack(Kraus_ops).reshape(k * d, d)

        # 1) Create a random matrix of shape (k*d, d):
        if dtype == torch.float64:
            M = torch.randn(k * d, d, dtype=torch.float64, device=device)
        else:
            real_part = torch.randn(k * d, d, dtype=torch.float64, device=device)
            imag_part = torch.randn(k * d, d, dtype=torch.float64, device=device)
            M = real_part + 1j * imag_part
        M = M.to(dtype)  # ensure dtype

        # 2) Perform a QR factorization to get an isometry Q.
        Q, R = torch.linalg.qr(M)
        # 3) Partition Q into k blocks, each of dimension (d, d).
        Kraus_ops = []
        for i in range(k):
            K_i = Q[i * d : (i + 1) * d, :]
            Kraus_ops.append(K_i)
        return torch.stack(Kraus_ops).reshape(k * d, d)


class MPSTPCP(nn.Module):
    def __init__(self, N, K, d=2, with_identity: bool = False, manifold: ManifoldType = ManifoldType.CANONICAL):
        """
        Args:
            N (int): Number of density matrices (batch size).
            d (int): Local dimension for each qubit (usually 2).
                     For a 2-qubit system, the total dimension is d^2.
        """
        super().__init__()
        self.N = N
        self.K = K
        self.d = d  # single-qubit dimension (d=2)

        self.L = N - 1
        self.kraus_ops = kraus_operators(K, L=self.L, with_identity=with_identity, manifold=manifold)
        self.manifold = self.kraus_ops.manifold

    def forward(self, X, normalize: bool = True):
        """
        Args:
            X (tensor): shape (batch_size, N, 2), where N is the number of qubits (or pixels).
        """
        assert X.shape[1:] == (self.N, 2), f"Expected X to have shape (batch_size, {self.N}, 2), but got {X.shape}"
        if normalize:
            X = X / torch.norm(X, dim=-1).unsqueeze(-1)

        # self.proj_stiefel(check_on_manifold=True)

        batch_size = X.shape[0]
        self.rhos = []
        rho1 = self.get_rho(X[:, 0])
        rho2 = self.get_rho(X[:, 1])
        init_rho = self.tensor_product(rho1, rho2)

        rho = init_rho
        for i in range(self.L):
            kraus_ops = self.kraus_ops[i].reshape(
                self.K, self.kraus_ops.act_size, self.kraus_ops.act_size
            )
            rho = self.forward_layer(rho, kraus_ops)
            self.rhos.append(rho.detach().clone())

            if i < self.L - 1:
                rho = self.partial(rho, 0)
                next_rho = self.get_rho(X[:, i + 2])
                rho = self.tensor_product(rho, next_rho)

            # Start of Selection
            rho_out = self.partial(rho, 0)

            # rho_test = rho_out[0, :, :]
            # trace = torch.trace(rho_test)
            # assert torch.isclose(trace, torch.tensor(1.0, device=rho_test.device, dtype=rho_test.dtype)), f"Trace of rho_out is {trace}, expected 1."
            # diag = torch.diagonal(rho_test, dim1=-2, dim2=-1)
            # assert torch.all(diag >= 0), "Some diagonal elements of rho_out are negative."

        self.rhos = torch.stack(self.rhos)
        return rho_out[..., 0, 0].reshape(batch_size)

    @staticmethod
    def tensor_product(rho1, rho2):
        batch_size = rho1.shape[0]
        d = rho1.shape[1]
        return torch.einsum("bij,bkl->bikjl", rho1, rho2).reshape(batch_size, d**2, d**2)

    @property
    def params(self):
        return list(self.kraus_ops.parameters())
    
    @staticmethod
    def get_rho(x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 2), "x must be a tensor of shape (batch_size, 2)"
        return torch.einsum("bi,bj->bij", x, x.conj())
    
    def forward_layer(self, rho, kraus_ops):
        """
        Applies a layer of Kraus operators to the input density matrix `rho`.

        This function projects the Kraus operators onto the Stiefel manifold
        to ensure they remain valid quantum operations. It then applies these
        operators to the input density matrix `rho` to produce a new density
        matrix.

        Parameters:
        - rho (torch.Tensor): A batch of density matrices with shape 
          (batch_size, d^2, d^2), where `d` is the dimension of the quantum 
          system.
        - kraus_ops (torch.Tensor): A set of Kraus operators with shape 
          (K, d, d), where `K` is the number of Kraus operators.

        Returns:
        - new_rho (torch.Tensor): The resulting batch of density matrices 
          after applying the Kraus operators, with shape (batch_size, d, d).
        """
        self.proj_stiefel(check_on_manifold=True)
        batch_size = rho.shape[0]
        assert rho.shape == (batch_size, self.d**2, self.d**2), "rho must be a tensor of shape (batch_size, d^2, d^2)"

        # kraus_ops shape: (K, d, d)
        kraus_ops_dagger = kraus_ops.conj().transpose(-1, -2)  # K_i^\dagger

        partial = kraus_ops.unsqueeze(1) @ rho.unsqueeze(0)  # (K, N, d, d)
        out = partial @ kraus_ops_dagger.unsqueeze(1)         # (K, N, d, d)
        new_rho = out.sum(dim=0)  # Sum over the Kraus index K => (N, d, d)
        return new_rho

    def partial(self, rho, site):
        """
        Perform a partial trace over one qubit (site=0 or site=1)
        for a batch of two-qubit states.
        
        - rho.shape = (N, d^2, d^2)   # e.g. d^2=4 for 2 qubits
        - site=0 means trace out qubit 0
        - site=1 means trace out qubit 1

        Returns: a batch of single-qubit density matrices with shape (N, d, d).
                 (Here, d=2.)
        """
        batch_size = rho.shape[0]
        assert rho.shape == (batch_size, self.d**2, self.d**2), f"rho must be a tensor of shape (batch_size, {self.d**2}, {self.d**2})"

        # Reshape => (N, d, d, d, d)
        rho_reshaped = rho.reshape(batch_size, self.d, self.d, self.d, self.d)

        if site == 0:
            reduced = torch.einsum("n a c a d -> n c d", rho_reshaped)
        elif site == 1:
            reduced = torch.einsum("n a c b c -> n a b", rho_reshaped)
        else:
            raise ValueError("site must be 0 or 1 for a 2-qubit system.")

        return reduced
    
    def proj_stiefel(self, check_on_manifold: bool = True):
        """
        Projects the Kraus operators onto the Stiefel manifold
        to ensure they remain valid quantum operations.
        """
        if check_on_manifold:
            if not self.check_point_on_manifold():
                for param in self.kraus_ops.parameters():
                    param.data.copy_(self.manifold.projx(param.data))
        else:
            for param in self.kraus_ops.parameters():
                param.data.copy_(self.manifold.projx(param.data))
    
    def check_point_on_manifold(self, rtol = 1e-5) -> bool:
        """
        Checks if the Kraus operators are on the Stiefel manifold.
        """
        for param in self.kraus_ops.parameters():
            if not self.manifold.check_point_on_manifold(param.data, rtol = rtol):
                return False
        return True
