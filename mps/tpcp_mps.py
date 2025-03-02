import torch  # do not change the order of import. otherwise, it causes error at QR factorization.
import opt_einsum as oe
from torch import nn
import numpy as np
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group
import geoopt  # Added for Geoopt functionalities
from enum import Enum
from mps.simple_mps import SimpleMPS
from mps.trainer import utils

class ManifoldType(Enum):
    CANONICAL = "canonical"
    FROBENIUS = "frobenius"
    EXACT = "exact"


class kraus_operators(nn.Module):

    K: int  # number of kraus operators.
    d: int = 2
    n: int = 2  # number of qubits it acts on.
    L: int  # number of kraus_operator layers.

    def __init__(self, 
                K, 
                L, 
                with_identity: bool = False, 
                init_with: torch.Tensor | None = None, 
                manifold: ManifoldType = ManifoldType.CANONICAL):
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
    def __init__(
        self, 
        N, 
        K, 
        d=2, 
        enable_r: bool = True,
        with_identity: bool = False, 
        manifold: ManifoldType = ManifoldType.CANONICAL
    ):
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
        self.with_probs = enable_r
        self.L = N - 1

        self.kraus_ops = kraus_operators(K, L=self.L, with_identity=with_identity, manifold=manifold)
        self.manifold = self.kraus_ops.manifold

        self.r = nn.Parameter(torch.eye(self.d, dtype=torch.float64, device=self.kraus_ops[0].device, requires_grad=True))

        self.pros0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.float64, device=self.kraus_ops[0].device)
        self.pros1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.float64, device=self.kraus_ops[0].device)

        self.initialize_W(random_init=False)

    def forward(self, X, normalize: bool = True, return_probs: bool = False, return_reg: bool = False):
        """
        Args:
            X (tensor): shape (batch_size, N, 2), where N is the number of qubits (or pixels).
        """
        assert X.shape[1:] == (self.N, 2), f"Expected X to have shape (batch_size, {self.N}, 2), but got {X.shape}"
        if normalize:
            X = X / torch.norm(X, dim=-1).unsqueeze(-1)
        

        # normalize r and W
        r = self.r / (torch.norm(self.r) / np.sqrt(self.r.shape[0]))



        # self.proj_stiefel(check_on_manifold=True, print_log=True)

        batch_size = X.shape[0]
        rho1 = self.get_rho(X[:, 0])
        rho2 = self.get_rho(X[:, 1])
        init_rho = self.tensor_product(rho1, rho2)

        rho = init_rho
        log_sr_list = []
        for i in range(self.L):
            kraus_ops = self.kraus_ops[i].reshape(
                self.K, self.kraus_ops.act_size, self.kraus_ops.act_size
            )
            rho = self.forward_layer(rho, kraus_ops)

            if i < self.L - 1:
                rho, log_sr = self.partial(rho, 0, self.W[i])
                log_sr_list.append(log_sr.mean())
                next_rho = self.get_rho(X[:, i + 2])
                rho = self.tensor_product(rho, next_rho)

            # Start of Selection
        rho_out, log_sr = self.partial(rho, 0, self.W[self.L - 1])
        log_sr_list.append(log_sr.mean())

        self.rho_last = rho_out.detach().clone()

        log_sr_pq = sum(log_sr_list) / len(log_sr_list)

        mes0 = r @ self.pros0 @ r.T.conj()
        mes0_result = torch.einsum("ij,...ji->...", mes0, rho_out)


        mes1 = r @ self.pros1 @ r.T.conj()
        mes1_result = torch.einsum("ij,...ji->...", mes1, rho_out)
        outputs = torch.stack([mes0_result, mes1_result], dim=-1)
        probs = self._to_probs(outputs)
        if return_probs:
            return probs if not return_reg else (probs, log_sr_pq)
        else:
            return probs[:, 0] if not return_reg else (probs[:, 0], log_sr_pq)

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
        # self.proj_stiefel(check_on_manifold=True)
        batch_size = rho.shape[0]
        assert rho.shape == (batch_size, self.d**2, self.d**2), "rho must be a tensor of shape (batch_size, d^2, d^2)"

        # kraus_ops shape: (K, d, d)
        kraus_ops_dagger = kraus_ops.conj().transpose(-1, -2)  # K_i^\dagger

        partial = kraus_ops.unsqueeze(1) @ rho.unsqueeze(0)  # (K, N, d, d)
        out = partial @ kraus_ops_dagger.unsqueeze(1)         # (K, N, d, d)
        new_rho = out.sum(dim=0)  # Sum over the Kraus index K => (N, d, d)
        return new_rho

    def partial(self, rho: torch.Tensor, site: int, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a partial trace over one qubit (site=0 or site=1)
        for a batch of two-qubit states, possibly *weighted* by `weight`.

        - rho.shape = (batch_size, d^2, d^2)   # e.g. d^2=4 if 2 qubits
        - site = 0 or 1 indicates which qubit to trace out.
        - weight (torch.Tensor or None):
            If None, do the usual partial trace.
            If not None, must be shape (2,) with sum(weight) = 1, 
            and we multiply each 'component' of the partial trace by weight[a].
        Returns:
            reduced_rho of shape (batch_size, d, d), i.e. a single-qubit density matrix.
            reg of shape (batch_size,), i.e. the regularization term.
        """
        batch_size = rho.shape[0]
        assert rho.shape == (batch_size, self.d**2, self.d**2), (
            f"rho must be (batch_size, {self.d**2}, {self.d**2}), got {rho.shape}"
        )

        # assert torch.allclose(torch.norm(weight), torch.tensor(1.0, dtype=weight.dtype, device=weight.device), atol=1e-10), f"weight must sum to 1, got {weight}"
        # assert torch.isclose(weight.norm(), torch.tensor(1.0, dtype=weight.dtype, device=weight.device)), f"weight must sum to 1, got {weight}"
        if weight is None:
            weight = torch.ones(self.d, dtype=rho.dtype, device=rho.device)
        
        # set the maximum element of each row to 1
        weight_prob = torch.abs(weight)
        weight_prob = weight_prob / torch.max(weight_prob)
        # assert np.isclose(np.linalg.norm(weight.detach().cpu().numpy()), 1.0, atol=1e-10), f"weight must sum to 1, got {weight}"

        # Reshape => (batch_size, d, d, d, d)
        # We can call these indices: (n, a, b, c, d).
        rho_reshaped = rho.reshape(batch_size, self.d, self.d, self.d, self.d)

        # The idea:
        #   * site=0 => trace out the first qubit index => "n a c a d -> n c d"
        #   * site=1 => trace out the second qubit index => "n a c b c -> n a b"
        # Weighted partial trace means we multiply the summand by weight[a] or weight[b].

        if site == 0:
            # Weighted partial trace
            # "n a c a d, a -> n c d"
            # i.e. sum over 'a' but multiply each slice by weight[a]
            assert weight_prob.shape == (2,), f"weight must be shape (2,), got {weight_prob.shape}"
            reduced = torch.einsum("n a c a d, a->n c d", rho_reshaped, weight_prob)

        elif site == 1:
            # Weighted partial trace
            # multiply each slice by weight[b]
            # We'll follow the same index pattern: "n a c b c, b->n a c"
            # then rename 'c' -> 'b' so shape is (n, a, b).
            assert weight_prob.shape == (2,), f"weight must be shape (2,), got {weight_prob.shape}"
            reduced = torch.einsum("n a c b c, c->n a b", rho_reshaped, weight_prob)
            reduced = reduced.reshape(batch_size, self.d, self.d)
        else:
            raise ValueError("site must be 0 or 1 for a 2-qubit system.")

        # print(reduced.shape)
        success_rate = torch.einsum("nii->n", reduced)
        return reduced / success_rate.unsqueeze(-1).unsqueeze(-1), -torch.log(success_rate)

    def initialize_W(self, init_with: torch.Tensor | None = None, random_init: bool = False):
        if init_with is not None:
            self.W = nn.Parameter(init_with, requires_grad=True)
        else:
            if random_init:
                self.W = nn.Parameter(torch.randn(self.L, 2, dtype=torch.float64), requires_grad=True)
            else:
                self.W = nn.Parameter(torch.ones(self.L, 2, dtype=torch.float64), requires_grad=True)
        
        # set the maximum element of each row to 1
        self.W.data[:] /= torch.max(self.W.data, dim=1, keepdim=True).values
    
    def proj_stiefel(self, check_on_manifold: bool = True, print_log: bool = False, rtol: float = 1e-5):
        """
        Projects the Kraus operators onto the Stiefel manifold
        to ensure they remain valid quantum operations.
        """
        if check_on_manifold:
            if not self.check_point_on_manifold(rtol = rtol, print_log = print_log):
                for param in self.kraus_ops.parameters():
                    param.data.copy_(self.manifold.projx(param.data))
        else:
            for param in self.kraus_ops.parameters():
                param.data.copy_(self.manifold.projx(param.data))
    
    def check_point_on_manifold(self, rtol = 1e-5, print_log: bool = False) -> bool:
        """
        Checks if the Kraus operators are on the Stiefel manifold.
        """
        for param in self.kraus_ops.parameters():
            ok, reason = self.manifold.check_point_on_manifold(param.data,explain=True, rtol = rtol)
            if not ok:
                if print_log:
                    print(f"Kraus operator is not on the Stiefel manifold: {reason}")
                return False
        return True

    @staticmethod
    def embed_isometry(v):
        """
        v is a vector in R^d


        convert isometry to unitary
        ┌───┐
        * -│   │
        * -│ V │- * 
        └───┘

        ┌───┐
        * -│   │- |0>
        * -│U  │- * 
        └───┘
        """
        assert len(v.shape) == 3
        d = v.shape[0]
        i = torch.einsum("abc, abd->cd", v, v)
        assert torch.allclose(i, torch.eye(d, dtype=v.dtype, device=v.device))
        V = torch.empty((d, d, d, d), dtype=v.dtype, device=v.device)
        V[:, :, 0, :] = v
        Q,R = torch.linalg.qr(V.reshape(d**2, d**2), mode="complete")
        Q = Q.reshape(d, d, d, d)

        if not torch.allclose(Q[:, :, 0, :], v, atol=1e-5):
            raise ValueError("QR decomposition failed")
        else:
            return Q

    def set_canonical_mps(self, smps: SimpleMPS):
        r"""
        Convert an existing SimpleMPS (with N sites, chi=d=2) into
        a sequence of 2-qubit unitaries (one per MPS bond), and store
        these unitaries as Kraus operators (K=1) in this TPCP model.

        This closely follows the logic of `_set_canonical_mps` in the
        HuMPS class, but adapts it for TPCP usage:

        1) call `smps.mps.convert_to_canonical()`,
        2) build isometries for the interior bonds,
        3) merge the left boundary and right boundary via einsum & QR,
        4) reshape each resulting rank-4 tensor (2,2,2,2) into (4,4),
        5) unitarize via QR,
        6) copy into self.kraus_ops[l].

        *Assumes* self.K == 1, self.L == N-1, d=2, chi=2, etc.
        """
        # 1) Basic dimension checks
        if smps.N != self.N:
            raise ValueError(f"MPS N={smps.N}, but TPCP N={self.N}.")
        if smps.d != 2 or smps.chi != 2:
            raise ValueError("This method assumes d=2, chi=2.")
        if self.K != 1:
            raise NotImplementedError("Only supports K=1 (unitary channel).")

        # 2) Convert the MPS to canonical form
        #    Typically returns a list of N tensors, e.g. for N=4:
        #    params[0] ~ (1, 2, 2),  params[-1] ~ (2, 2, 1), middle ones ~ (2, 2, 2).
        params = smps.mps.convert_to_canonical()
        if len(params) != self.N + 1: # +1 include for the target state
            raise RuntimeError(f"Expected {self.N + 1} canonical tensors, got {len(params)}.")
        
        MPS_list = [self.embed_isometry(m.clone()) for m in params[1:-1]]
        uU = torch.einsum("ak,kbcd->abcd",params[0], MPS_list[0])
        MPS_list[0] = uU.clone()
        h = params[-1] 
        # H = h @ h.conj().T
        # e, V = torch.linalg.eigh(H)
        q, r = torch.linalg.qr(h)
        U = MPS_list[-1]
        U = torch.einsum("abcd, de -> abce", U, q)
        MPS_list[-1] = U.clone()

        for i in range(self.L):
            self.kraus_ops[i].data[:] = MPS_list[i].clone().reshape(self.d**2, self.d**2).T

        assert self.check_point_on_manifold(rtol = 1e-5), "Kraus operators are not on the Stiefel manifold"

        # self.mes.data[:] = r @ self.mes @ r.conj().T
        if not self.with_probs:
            self.r.data[:] = r

        # if use_r:
        #     self.mes.data[:] = r @ self.mes @ r.conj().T
        # else:
        #     # calculate QR 
        #     q, r = torch.linalg.qr(r)
        #     unflip = torch.linalg.diagonal(r).sign().add(0.5).sign()
        #     q *= unflip[..., None, :]
        #     self.mes.data[:] = q @ self.mes @ q.conj().T
    
    def normalize_w_and_r(self):
        with torch.no_grad():
            self.W.data[:] = torch.abs(self.W.data)
            self.W.data[:] /= torch.max(self.W.data, dim=1, keepdim=True).values
            self.r.data[:] = self.r / torch.linalg.norm(self.r)
    
    @staticmethod
    def _to_probs(outputs):
        """
        Convert model outputs to probabilities.
        """
        return utils.to_probs(outputs)
    


def regularize_weight(w, p = 4):
    """
    Computes a regularization term based on the log of the ratio Var(W)/E(W)
    of the post-selection success rate when the input state is uniform.
    
    Here, W = ∏_i w_i(x_i) is a product of weight functions evaluated on 
    uniformly distributed bit outcomes (with each bit taking 0 or 1 with probability 1/2).
    
    For each layer i (with weight vector w_i = [w_i(0), w_i(1)]), define:
        μ_i = (w_i(0) + w_i(1)) / 2
        ν_i = (w_i(0)^2 + w_i(1)^2) / 2
    
    Then:
        E[W]  = ∏_i μ_i,
        E[W^2] = ∏_i ν_i,
        Var(W) = E[W^2] - (E[W])^2.
    
    A compact expression for the ratio is:
        Var(W)/E[W] = ∏_i ( (w_i(0)^2+w_i(1)^2) / (w_i(0)+w_i(1)) ) - ∏_i ((w_i(0)+w_i(1))/2)
    
    Args:
        w (torch.Tensor): Tensor of shape (L, 2), where L is the number of layers.
        eps (float): A small number added for numerical stability.
        
    Returns:
        torch.Tensor: A scalar representing log(Var(W)/E[W] + eps).
    """

    assert p > 2, "p must be greater than 2"

    L = w.shape[0]
    if w.shape != (L, 2):
        raise ValueError(f"Expected w to have shape ({L}, 2), but got {w.shape}.")
    
    # normalize w 
    w = w / torch.norm(w, dim=1, keepdim=True)
    # if not torch.allclose(torch.norm(w, dim=1), torch.ones(L, device=w.device, dtype=w.dtype), atol=1e-6):
    #     raise ValueError("The norm of each row in w must be 1.")
    
    w_p = w**p

    return w_p.sum(dim=1).mean() - 1/2
    
    
    
    
    
