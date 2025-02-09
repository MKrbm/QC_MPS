import torch # do not change the order of import. otherwise, it causes error at QR factorization.
import opt_einsum as oe
from torch import nn
import numpy as np
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group


class kraus_operators(nn.Module):

    K: int  # number of kraus operators.
    d: int = 2
    n: int = 2  # number of qubits it acts on.
    L: int  # number of kraus_operator layers.

    def __init__(self, K, L, with_identity: bool = False, init_with: torch.Tensor | None = None):
        super().__init__()
        self.K = K
        self.L = L

        self.act_size = self.d ** self.n
        self.kraus_ops = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.K * self.act_size, self.act_size, dtype=torch.float64)) for _ in range(self.L)]
        )
        self.init_params(init_with=init_with, init_with_identity=with_identity)

    def __getitem__(self, idx):
        return self.kraus_ops[idx]

    def init_params(self, init_with: torch.Tensor | None = None, init_with_identity: bool = False):
        """
        Initialize or overwrite the parameters of the Kraus operators.

        Args:
            init_with (torch.Tensor, optional): Tensor to initialize Kraus operators with.
                                                Must represent a TPCP map if provided.
        """
        if init_with is not None:
            if len(init_with) != len(self.kraus_ops):
                print(f"init_with.shape: {len(init_with)}, self.kraus_ops.shape: {len(self.kraus_ops)}")
                raise ValueError(f"init_with tensor must have shape {self.kraus_ops}")
            for l in range(self.L):
                if not self.is_tpcp(init_with[l]):
                    raise ValueError("The provided tensor does not represent a TPCP map.")
                self.kraus_ops[l].data[:] = init_with[l].clone()
        else:
            for l in range(self.L):
                kraus_ops = self.random_tpcp_map_torch(
                    self.act_size, self.K, init_with_identity
                )
                self.kraus_ops[l].data[:] = kraus_ops.reshape(self.K * self.act_size, self.act_size)

    @staticmethod
    def is_tpcp(kraus_tensor: torch.Tensor) -> bool:
        """
        Check if the given Kraus tensor represents a TPCP map.

        Args:
            kraus_tensor (torch.Tensor): Tensor containing Kraus operators
                                         with shape (K*d, d).

        Returns:
            bool: True if the tensor is TPCP, False otherwise.
        """
        K, d, d = kraus_tensor.shape
        kraus = kraus_tensor
        identity = torch.eye(d, dtype=kraus.dtype, device=kraus.device)
        sum_kd = torch.stack([K_i.conj().transpose(-1, -2) @ K_i for K_i in kraus]).sum(dim=0)
        return torch.allclose(sum_kd, identity, atol=1e-6)

    @staticmethod
    def random_tpcp_map_torch(
        d, k, with_identity=False, dtype=torch.float64, device="cpu"
    ) -> torch.Tensor:
        """
        Generate a random TPCP map in d-dimensional Hilbert space
        with k Kraus operators using the isometry method, in PyTorch.

        Args:
            d (int): Hilbert space dimension.
            k (int): Number of Kraus operators.
            with_identity (bool): Whether to include the identity operator.
            dtype (torch.dtype): Complex dtype to use (complex64 or complex128).
            device (str or torch.device): Device for tensors (e.g. 'cpu' or 'cuda').

        Returns:
            A tensor representing k Kraus operators stacked together with shape (k*d, d).
        """
        if with_identity:
            identity = torch.eye(d, dtype=dtype, device=device)
            Kraus_ops = [identity for _ in range(k)]
            return torch.stack(Kraus_ops).reshape(k * d, d)

        # 1) Create a random complex matrix of shape (k*d, d):
        #    We'll sample real and imaginary parts from a normal distribution.
        if dtype == torch.float64:  
            real_part = torch.randn(k * d, d, dtype=torch.float64, device=device)
            M = real_part


        else:
            real_part = torch.randn(k * d, d, dtype=torch.float64, device=device)
            imag_part = torch.randn(k * d, d, dtype=torch.float64, device=device)
            M = real_part + 1j * imag_part
        M = M.to(dtype)  # ensure complex dtype

        # 2) Perform a QR factorization to get an isometry Q
        #    (Q is (k*d, d), R is (d, d)).
        Q, R = torch.linalg.qr(M)
        # 3) Partition Q into k blocks, each of dimension (d, d).
        Kraus_ops = []
        for i in range(k):
            # Slice rows [i*d : (i+1)*d] to form K_i
            K_i = Q[i * d : (i + 1) * d, :]
            Kraus_ops.append(K_i)

        return torch.stack(Kraus_ops).reshape(k * d, d)


class MPSTPCP(nn.Module):
    def __init__(self, N, K, d=2, with_identity: bool = False):
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
        self.kraus_ops = kraus_operators(K, L=self.L, with_identity=with_identity)

    def forward(self, X, normalize: bool = True):
        """
        Args:
            X (tensor): shape (batch_size, N, 2), where N is the number of qubits (or pixels).

        """

        assert X.shape[1:] == (
            self.N,
            2,
        ), f"Expected X to have shape (batch_size, {self.N}, 2), but got {X.shape}"
        if normalize:
            X = X / torch.norm(X, dim=-1).unsqueeze(-1)
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

            if i < self.N - 2:
                rho = self.partial(rho, 0)
                next_rho = self.get_rho(X[:, i + 2])
                rho = self.tensor_product(rho, next_rho)

        res = self.partial(rho, 0)[..., 0, 0]
        self.rhos = torch.stack(self.rhos)
        return res

    @staticmethod
    def tensor_product(rho1, rho2):
        batch_size = rho1.shape[0]
        d = rho1.shape[1]
        return torch.einsum("bij,bkl->bikjl", rho1, rho2).reshape(
            batch_size, d**2, d**2
        )
    @property
    def params(self):
        return list(self.kraus_ops.parameters())
    
    @staticmethod
    def get_rho(x):

        batch_size = x.shape[0]
        assert x.shape == (batch_size, 2), "x must be a tensor of shape (batch_size, 2)"

        return torch.einsum("bi,bj->bij", x, x.conj())

    def forward_layer(self, rho, kraus_ops):

        batch_size = rho.shape[0]
        assert rho.shape == (
            batch_size,
            self.d**2,
            self.d**2,
        ), "rho must be a tensor of shape (batch_size, d^2, d^2)"

        #  rho shape  => (N, d, d)
        #  kraus_ops => (K, d, d)

        # We'll compute  partial = K_i @ rho for all i,n at once using broadcasting:
        #   partial.shape => (K, N, d, d)
        # Explanation:
        #   kraus_ops.unsqueeze(1) => shape (K, 1, d, d)
        #   rho.unsqueeze(0)       => shape (1, N, d, d)
        # When we do matmul, PyTorch broadcasts (K,1,d,d) with (1,N,d,d) => (K,N,d,d).
        kraus_ops_dagger = kraus_ops.conj().transpose(-1, -2)  # K_i^\dagger

        partial = kraus_ops.unsqueeze(1) @ rho.unsqueeze(0)  # => (K, N, d, d)
        out = partial @ kraus_ops_dagger.unsqueeze(1)  # => (K, N, d, d)

        # Now sum over the Kraus index K to get shape (N, d, d)
        new_rho = out.sum(dim=0)  # sum over the 0th dim => (N, d, d)

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
        assert rho.shape == (
            batch_size,
            self.d**2,
            self.d**2,
        ), f"rho must be a tensor of shape (batch_size, {self.d**2}, {self.d**2})"

        # Reshape => (N, d, d, d, d)
        # Indices: rho[n, a, c, b, d] = element ((a,c),(b,d))
        rho_reshaped = rho.reshape(batch_size, self.d, self.d, self.d, self.d)

        if site == 0:
            # Trace out the first qubit => sum over a=b
            # Using einsum, indices to keep: n, c, d
            # We want: \sum_a rho[n, a, c, a, d]
            # => (N, d, d)
            reduced = torch.einsum("n a c a d -> n c d", rho_reshaped)

        elif site == 1:
            # Trace out the second qubit => sum over c=d
            # We want: \sum_c rho[n, a, c, b, c]
            # => (N, d, d)
            reduced = torch.einsum("n a c b c -> n a b", rho_reshaped)

        else:
            raise ValueError("site must be 0 or 1 for a 2-qubit system.")

        return reduced
