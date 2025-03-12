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

        # mps_shapes for the left, middle, and right cores:
        # First core: (d, chi_max)
        # Middle cores: (chi_max, d, chi_max)
        # Last core: (chi_max, d)
        self.mps_shapes = [(self.d, self.chi_max)] \
                          + [(self.chi_max, self.d, self.chi_max)] * (self.N - 1) \
                          + [(self.chi_max, self.d)]

        self.initialize_params(std=eps)
    
    def initialize_params(self, std=1e-2):
        MPS_list = []
        for i in range(len(self.mps_shapes)):
            if i == 0:
                # left-most core: shape (d, chi_max)
                core = torch.zeros(self.mps_shapes[i], dtype=self.dtype)
                # core[0] = 1
                # core[:] = torch.eye(self.d, dtype=self.dtype)
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            elif i == len(self.mps_shapes) - 1:
                # right-most core: shape (chi_max, d)
                core = torch.zeros(self.mps_shapes[i], dtype=self.dtype)
                core[:] = torch.eye(self.chi_max, dtype=self.dtype)
                # core += torch.normal(mean=0.0, std=std, size=core.shape)
            else:
                # middle cores: shape (chi_max, d, chi_max)
                core = torch.stack([torch.eye(self.chi_max, dtype=self.dtype)] * self.d).permute(1, 0, 2)
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            MPS_list.append(core)
        
        self.params = nn.ParameterList([nn.Parameter(mps) for mps in MPS_list])
    
    def get_canonical_form(self):
        """
        Convert the list of MPS cores (in numpy format) to a canonical form
        using successive SVD decompositions.
        
        The algorithm follows these steps:
            1. Reshape the first core to (1, d, chi_max)
            2. For each site (except the last):
               - Reshape the current block A to a matrix.
               - Perform SVD.
               - Normalize the singular values.
               - Reshape U back to a tensor and store it.
               - Contract R with the next core using einsum.
            3. Process the final core using the accumulated R.
            4. Reshape the first core back to (d, chi_max).
        
        Returns:
            new_MPS_list: list of numpy arrays in canonical form.
        """
        d = self.d
        chi_max = self.chi_max
        MPS_list = [core.detach().cpu().numpy() for core in self.params]

        # Work on a copy so that the original is preserved.
        MPS_list = [np.copy(core) for core in MPS_list]
        # Reshape the first core: from (d, chi_max) -> (1, d, chi_max)
        MPS_list[0] = MPS_list[0].reshape(1, d, chi_max)
        new_MPS_list = [np.zeros(core.shape) for core in MPS_list]

        # Initialize A as the first core
        A = np.copy(MPS_list[0])
        
        for i in range(len(MPS_list) - 1):
            shapeA = A.shape
            # Reshape A to merge the left indices with the physical index.
            A = A.reshape(A.shape[0] * d, -1)
            
            # Perform SVD
            U, S, V = np.linalg.svd(A, full_matrices=False)
            
            # Normalize S (ensuring the normalization remains consistent)
            S = S / np.linalg.norm(S)

            
            # Update chi (could be truncated if needed)
            chi_new = len(S)
            # Form the matrix R = diag(S) * V[:chi_new]
            R = np.diag(S) @ V[:chi_new]
            
            # Reshape U back to the tensor shape of the current core.
            At = U.reshape(shapeA[0], shapeA[1], -1)
            new_MPS_list[i][:At.shape[0], :At.shape[1], :At.shape[2]] = At
            
            if i < len(MPS_list) - 2:
                # Contract R with the next core.
                A = np.copy(MPS_list[i+1])
                A = np.einsum("ij, jdk->idk", R, A)
        
        # Process the final core.
        Af = np.copy(MPS_list[-1])
        Af = R @ Af
        new_MPS_list[-1][:] = Af
        
        # Reshape the first core back to its original shape: (d, chi_max)
        new_MPS_list[0] = new_MPS_list[0].reshape(d, chi_max)
        
        return new_MPS_list

    def get_canonical_form_v2(self):
        MPS_list = [core.detach().cpu().numpy() for core in self.params]

        MPS_list[0] = MPS_list[0].reshape(1, 2, 2)
        MPS_list[-1] = MPS_list[-1].reshape(2, 2, 1)
        mpstate = tn.FiniteMPS(MPS_list, canonicalize=True, center_position=len(MPS_list)-1)
        return [torch.tensor(t, dtype=self.dtype).reshape(s) for t, s in zip(mpstate.tensors, self.mps_shapes)]

    def convert_to_canonical(self):
        """
        Convert the current MPS parameters to their canonical form.
        This function:
            1. Extracts the parameters as numpy arrays.
            2. Calls get_canonical_form to perform the SVD-based canonicalization.
            3. Converts the canonical numpy arrays back into torch tensors.
        
        Returns:
            new_MPS_list_tensor: list of torch tensors in canonical form.
        """
        # Extract parameters to a list of numpy arrays.
        
        # Get the canonical form using the static method.
        new_MPS_list_np = self.get_canonical_form()
        
        # Convert the canonical numpy cores back to torch tensors.
        new_MPS_list_tensor = [
            torch.tensor(core, dtype=self.dtype, device=self.device)
            for core in new_MPS_list_np
        ]
        
        self.set_params(new_MPS_list_tensor)

    def set_params(self, params: list[torch.Tensor]):
        assert len(params) == len(self.params), (
            f"Number of parameters must match, but got {len(params)} and {len(self.params)}"
        )
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
    
    def rescale(self):
        """
        Rescales the MPS parameters to normalize the output magnitude.
        This is done by:
        1. Creating a normalized random test state
        2. Computing the system scale from MPS output
        3. Calculating the per-site scaling factor
        4. Rescaling all MPS parameters
        
        Returns:
            float: The original system scale before rescaling
        """
        # Create and normalize test state
        x = torch.randn(self.N, 2, dtype=self.mps.dtype, device=self.device)
        x = torch.abs(x)
        x = x / x.sum(dim=1).unsqueeze(1)
        
        # Calculate system scale
        out = self(x.reshape(self.N, 1, 2))
        sys_scale = torch.abs(out).sum().item()
        
        # Calculate per-site scaling factor
        s = sys_scale ** (1 / self.N)
        
        # Rescale parameters
        params = self.mps.params
        params_rescaled = [p.detach().clone() / s for p in params]
        
        # Update parameters
        self.mps.set_params(params_rescaled)
        self.initialize_MPS()
        
        return sys_scale

    def compress_chi(self, new_chi: int) -> 'SimpleMPS':
        """
        Compresses the MPS by truncating the bond dimensions to new_chi.
        This is a simple truncation method that:
        1. Converts the MPS to canonical form
        2. Rescales the MPS parameters
        3. Creates a new SimpleMPS instance with smaller bond dimension
        4. Truncates the parameters to the new bond dimension
        
        Args:
            new_chi (int): New bond dimension (must be smaller than current chi)
            
        Returns:
            SimpleMPS: A new instance with compressed bond dimension
        """
        # First convert to canonical form and rescale
        self.mps.convert_to_canonical()
        self.rescale()
        self.initialize_MPS()
        
        # Get current parameters
        params = self.mps.params
        
        # Create new SimpleMPS instance with smaller bond dimension
        new_mps = SimpleMPS(
            N=self.N,
            chi=new_chi,
            d=self.d,
            l=self.l,
            layers=self.layers,
            device=self.device,
            dtype=self.dtype,
            optimize=self.optimize,
            eps=1e-2  # You might want to make this configurable
        )
        
        # Initialize new parameters with correct shapes
        new_params = [torch.empty(shape, dtype=self.dtype, device=self.device) 
                     for shape in new_mps.mps.mps_shapes]
        
        # Truncate and copy parameters
        for i, (p, new_p) in enumerate(zip(params, new_params)):
            if i != 0 and i != len(params) - 1:
                # Middle tensors: (chi, d, chi) -> (new_chi, d, new_chi)
                new_p[:] = p.data[:new_chi, :, :new_chi]
            elif i == 0:
                # First tensor: (d, chi) -> (d, new_chi)
                new_p[:] = p.data[:, :new_chi]
            else:
                # Last tensor: (chi, d) -> (new_chi, d)
                new_p[:] = p.data[:new_chi, :]
        
        # Set the new parameters and initialize
        new_mps.mps.set_params(new_params)
        new_mps.initialize_MPS()
        
        return new_mps
