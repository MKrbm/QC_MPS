
import opt_einsum as oe
import torch
from torch import nn
import numpy as np
import math
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group
import tensornetwork as tn

from opt_einsum.testing import build_views
from opt_einsum.typing import PathType
from opt_einsum.contract import PathInfo, ContractExpression
import tensornetwork as tn
from tensornetwork.matrixproductstates.base_mps import BaseMPS

class uMPS(nn.Module):
    def __init__(
        self, 
        N: int, 
        chi: int, 
        d: int, 
        layers: int, 
        init_with_identity: bool = True,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        self.N = N
        self.chi = chi
        self.d = d
        self.layers = layers
        self.device = device
        self.dtype = dtype

        self.initialize_params(init_with_identity)
    
    def initialize_params(self, init_with_identity: bool = False):

        if self.dtype == torch.float64:
            MPS_unitaries = [ortho_group.rvs(self.chi ** 2).reshape((self.chi,) * 4) for _ in range((self.N - 1) * self.layers)]
        elif self.dtype == torch.complex128:
            MPS_unitaries = [unitary_group.rvs(self.chi ** 2).reshape((self.chi,) * 4) for _ in range((self.N - 1) * self.layers)]
        else:
            raise ValueError("Unsupported dtype")
        
        # later layers
        identity = np.eye(self.chi ** 2).reshape((self.chi,) * 4)
        # for l in range(1, self.layers):
        #     for _ in range(self.N - 1):
        #         MPS_unitaries.append(identity)
        
        if init_with_identity:
            for i in range(len(MPS_unitaries)):
                MPS_unitaries[i] = identity
        
        self.params = nn.ParameterList([
            nn.Parameter(torch.tensor(unitary, device=self.device, dtype=self.dtype)) for unitary in MPS_unitaries
        ])

    def set_unitaries(self, unitaries: list[torch.Tensor]):
        for i, unitary in enumerate(unitaries):
            self.params[i].data[:] = unitary
        
        self.check_unitary()
    
    def check_unitary(self):
        for i in range(len(self.params)):
            p = self.params[i].data.reshape(self.chi ** 2, self.chi ** 2)
            assert torch.allclose(p @ p.T.conj(), torch.eye(self.chi ** 2, dtype=self.dtype, device=self.device))

class MPS(nn.Module):
    def __init__(
        self, 
        N: int, 
        chi_max: int, 
        d: int, 
        eps: float = 1e-2,
        identity: bool = False,
        device: torch.device = torch.device("cpu"), 
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        self.N = N
        self.chi_max = chi_max
        self.d = d
        self.identity = identity
        self.device = device
        self.dtype = dtype

        self.initialize_params(std=eps)
    
    def initialize_params(self, std = 1e-2):

        self.mps_shapes = [(self.d, self.chi_max)] + [(self.chi_max, self.d, self.chi_max)] * (self.N - 1) + [(self.chi_max, self.d)]

        MPS_list = []
        for i in range(len(self.mps_shapes)):
            if i == 0:
                core = torch.zeros(self.mps_shapes[i], dtype=self.dtype)
                core[:, 0] = 1
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            elif i == len(self.mps_shapes)-1:
                core = torch.zeros(self.mps_shapes[i], dtype=self.dtype)
                core[0] = 1
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            else:
                core = torch.stack([torch.eye(self.chi_max, dtype=self.dtype)] * self.d).permute(1, 0, 2)
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            MPS_list.append(core)

        self.params = nn.ParameterList([
            nn.Parameter(mps) for mps in MPS_list
        ])
    def convert_to_canonical(self):
        MPS_list = [core.detach().cpu().numpy() for core in self.params]

        MPS_list[0] = MPS_list[0].reshape(1, 2, 2)
        MPS_list[-1] = MPS_list[-1].reshape(2, 2, 1)
        mpstate = tn.FiniteMPS(MPS_list, canonicalize=True, center_position=len(MPS_list)-1)
        return [torch.tensor(t, dtype=self.dtype).reshape(s) for t, s in zip(mpstate.tensors, self.mps_shapes)]
    
    def set_params(self, params: list[torch.Tensor]):
        assert len(params) == len(self.params), "Number of parameters must match, but got {} and {}".format(len(params), len(self.params))
        for i, param in enumerate(params):
            self.params[i].data[:] = param


class SimpleMPS(nn.Module):
    N: int
    chi: int # bond dimension (same as d)
    d: int # data input dimension (degree of freedom in each site)
    l: int # label dimension (normally same as d)
    layers: int # number of layers
    einsum_str: str
    n_tensors: int
    activation: bool
    optimize: str
    use_simple_path: bool
    path: PathType | None = None
    path_info: ContractExpression | None = None
    
    def __init__(
        self,
        N: int,
        chi: int,
        d: int,
        l: int,
        layers : int,
        optimize: str = "auto",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        seed: int = 0,
        eps: float = 1e-2
    ):
        super().__init__()
        # super().__init__(*args, **kwargs)
    
        assert d == l, "Data input dimension and class label dimension must be the same"

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.layers = layers
        self.N = N
        self.d = d
        self.l = l
        self.chi = chi
        self.optimize = optimize
        self.device = device
        self.dtype = dtype
        self.is_initialized = False
        self.Sz = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        self.mps = MPS(N, chi, d, eps=eps, identity=True, device=device, dtype=dtype)
        self.ops = []
        self.left_qubit_inds = []
        self.left_mps_inds = []
        self.size_dict = {}

        self.estr_mps = ""

        left_qubits = []
        ind = 1
        for i in range(self.N):
            q = oe.get_symbol(ind)
            self.estr_mps += "a{},".format(q)
            left_qubits.append(q)
            self.size_dict[q] = self.d
            self.left_qubit_inds.append(len(self.ops))
            self.ops.append(None)
            ind += 1
        
        #first layer

        q, b = left_qubits[0], oe.get_symbol(ind)
        self.estr_mps += "{}{},".format(q, b)
        self.left_mps_inds.append(len(self.ops))
        self.size_dict[b] = self.chi
        self.size_dict[q] = self.d
        self.ops.append(None)
        ind += 1

        #later layers
        for i in range(1, self.N):
            t, q, b = oe.get_symbol(ind-1), left_qubits[i], oe.get_symbol(ind)
            self.estr_mps += "{}{}{},".format(t, q, b)
            self.left_mps_inds.append(len(self.ops))
            self.size_dict[b] = self.chi
            self.size_dict[q] = self.d
            self.ops.append(None)
            ind += 1

        #last layer
        # the last qubit is for label
        q, ll = oe.get_symbol(ind-1), oe.get_symbol(ind)
        self.estr_mps += "{}{},".format(q, ll)
        self.left_mps_inds.append(len(self.ops))
        self.size_dict[q] = self.d
        self.size_dict[ll] = self.l
        ind += 1
        self.ops.append(None)

        self.estr_mps = self.estr_mps[:-1]
        self.estr_mps += "->a{}".format(ll)
        self.set_path_mps()
        self.initialize_MPS()
    
    def _get_path(self,einsum_str: str, d: int, optimize: str):
        unique_inds = set(einsum_str) - {',', '-', '>'}
        einsum_str = einsum_str.replace("a", "")
        einsum_str = einsum_str.replace("->", "")

        views = build_views(einsum_str, self.size_dict)
        path, path_info = oe.contract_path(einsum_str, *views, optimize=optimize)
        return path, path_info

    def set_path_mps(self, optimize: str = 'greedy'):
        if self.path is not None:
            print(f"Path is already set, finding the {optimize} path...")
        else:
            print("Path is not set, setting...")
        self.path, self.path_info = self._get_path(self.estr_mps, self.d, optimize)
        print("Found the path")
    

    def initialize_MPS(self):
        for i, ind in enumerate(self.left_mps_inds):
            self.ops[ind] = self.mps.params[i]
        print("Initialized MPS with random matrices")

    def forward(self, X: torch.Tensor):
        X = X.to(self.device).to(self.dtype)
        assert X.shape[0] == self.N, "Number of qubits must match"
        for i in range(self.N):
            self.ops[self.left_qubit_inds[i]] = X[i]
        # measurement qubits
        return torch.abs(oe.contract(self.estr_mps, *self.ops, backend="torch", optimize=self.path))

    def compressed(self, new_chi: int):
        """
        Create a new instance of SimpleMPS with compressed (lower bond dimension)
        MPS parameters using SVD-based truncation with einsum contractions.
        
        Even if mathematically the truncation gives a rank r < new_chi,
        we force the bond dimension to be new_chi by padding the extra elements with zeros.
        
        The original model remains unchanged.
        
        Returns:
            new_instance (SimpleMPS): A new SimpleMPS instance with bond dimension new_chi.
        """
        old_params = self.mps.params  # original list of MPS tensors
        new_params = []

        # --- Compress the first tensor ---
        # Assume the first tensor A0 has shape (d, old_chi)
        A0 = old_params[0]
        d = A0.shape[0]
        U, S, Vh = torch.linalg.svd(A0, full_matrices=False)
        # Mathematically, we would have r0 = min(new_chi, S.shape[0])
        r0 = min(new_chi, S.shape[0])
        # Truncate U to (d, r0)
        U_trunc = U[:, :r0]
        # Pad U_trunc to (d, new_chi) by filling the remaining columns with zeros
        new_A0 = torch.zeros(d, new_chi, dtype=A0.dtype, device=A0.device)
        new_A0[:, :r0] = U_trunc
        new_params.append(new_A0)

        # Build the bond tensor B = diag(S) @ Vh using einsum.
        # B is initially of shape (r0, old_chi)
        B = torch.einsum('i,ij->ij', S[:r0], Vh[:r0, :])
        # Pad B to shape (new_chi, old_chi)
        old_bond = B.shape[1]
        B_padded = torch.zeros(new_chi, old_bond, dtype=B.dtype, device=B.device)
        B_padded[:r0, :] = B
        B = B_padded

        # --- Compress intermediate tensors ---
        # For tensors 1 to N-2, assume each tensor has shape (old_chi, d, old_chi)
        for i in range(1, self.N):
            Ai = old_params[i]
            # Contract B (shape: (r_prev, old_chi)) with Ai (shape: (old_chi, d, old_chi))
            # Result T has shape: (r_prev, d, old_chi)
            T = torch.einsum('ab,bpg->apg', B, Ai)
            r_prev, d_val, chi_right = T.shape

            # Reshape T into a matrix for SVD: shape (r_prev * d, chi_right)
            T_mat = T.reshape(r_prev * d_val, chi_right)
            U, S, Vh = torch.linalg.svd(T_mat, full_matrices=False)
            r_new = min(new_chi, S.shape[0])
            U_trunc = U[:, :r_new]   # shape: (r_prev*d, r_new)
            S_trunc = S[:r_new]        # shape: (r_new,)
            Vh_trunc = Vh[:r_new, :]    # shape: (r_new, chi_right)

            # Reshape U_trunc back into a 3-index tensor: (r_prev, d, r_new)
            A_temp = U_trunc.reshape(r_prev, d_val, r_new)
            # Pad A_temp to shape (r_prev, d, new_chi)
            new_Ai = torch.zeros(r_prev, d_val, new_chi, dtype=A_temp.dtype, device=A_temp.device)
            new_Ai[:, :, :r_new] = A_temp
            new_params.append(new_Ai)

            # Update bond tensor B: B_new is shape (r_new, chi_right)
            B_new = torch.einsum('i,ij->ij', S_trunc, Vh_trunc)
            # Pad B_new to shape (new_chi, chi_right)
            B_padded = torch.zeros(new_chi, chi_right, dtype=B_new.dtype, device=B_new.device)
            B_padded[:r_new, :] = B_new
            B = B_padded

        # --- Compress the last tensor ---
        # Assume the last tensor has shape (old_chi, l)
        A_last = old_params[-1]
        # Contract the final bond tensor B (shape: (r_last, old_chi)) with A_last (shape: (old_chi, l))
        T = torch.einsum('ab,bl->al', B, A_last)  # resulting shape: (r_last, l)
        r_last, l_dim = T.shape
        # Pad T to shape (new_chi, l)
        T_padded = torch.zeros(new_chi, l_dim, dtype=T.dtype, device=T.device)
        T_padded[:r_last, :] = T
        T = T_padded
        new_params.append(T)

        # --- Create a new instance ---
        new_instance = SimpleMPS(
            N=self.N,
            chi=new_chi,
            d=self.d,
            l=self.l,
            layers=self.layers,
            optimize=self.optimize,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            eps=1e-2  # or use self.mps.eps if stored
        )
        
        # Replace its MPS parameters with our compressed parameters.
        new_instance.mps.set_params(new_params)
        new_instance.initialize_MPS()

        print(f"Created a new compressed MPS instance with bond dimension {new_chi}")
        return new_instance
