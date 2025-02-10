import opt_einsum as oe
import torch
from torch import nn
import numpy as np
import math
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group
import tensornetwork as tn

from opt_einsum.helpers import build_views
from opt_einsum.contract import PathInfo, ContractExpression
import tensornetwork as tn
from tensornetwork.matrixproductstates.base_mps import BaseMPS



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
    path: None = None
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
        with_identity: bool = True  # Added parameter
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
        self.init_mps(with_identity)  # Changed from initialize_MPS to init_mps
    
    def init_mps(self, with_identity: bool = True):
        """
        Initialize the MPS with optional identity initialization.
        
        Args:
            with_identity (bool): If True, initialize MPS with identity matrices. Otherwise, use random initialization.
        """
        self.mps = MPS(self.N, self.chi, self.d, with_identity, self.device, self.dtype)
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


class HuMPS(nn.Module):
    N: int
    chi: int # bond dimension (same as d)
    d: int # data input dimension (degree of freedom in each site)
    l: int # label dimension (normally same as d)
    layers: int # number of layers
    einsum_str: str
    n_tensors: int
    label_indicies: list[int]
    activation: bool
    optimize: str
    use_simple_path: bool
    path: None = None
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
        with_identity: bool = True  # Added parameter
    ):
        super().__init__()
        # super().__init__(*args, **kwargs)
    

        assert d == l == chi, "Data input dimension and class label dimension must be the same"

        self.layers = layers
        self.N = N
        self.chi = chi
        self.d = d
        self.l = l
        self.scale = 1.0
        self.optimize = optimize
        self.device = device
        self.dtype = dtype
        self.is_initialized = False
        self.umps = uMPS(N, chi, d, layers, with_identity, device, dtype)  # Passed with_identity
        self.projection = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)
        projector0 = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)
        projector1 = torch.tensor([[0, 0], [0, 1]], dtype=self.dtype, device=self.device)
        self.projector = torch.stack([projector0, projector1])
        self.eye = torch.eye(self.chi, dtype=self.dtype, device=self.device)
        self.r_tensor = self.eye.clone()
        self.r_tensor.requires_grad = True

        self.Sz = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)

        estr = ""
        left_qubits = [oe.get_symbol(i + 1) for i in range(N)]
        right_qubits = [oe.get_symbol(i + 1) for i in range(N, 2*N)]

        self.left_qubits_inds = [i for i in range(N)]
        self.right_qubits_inds = [i for i in range(N, 2*N)]

        self.left_mps_inds = []
        self.right_mps_inds = []

        self.size_dict = {}

        for s in left_qubits:
            estr += f",a{s}"
            self.size_dict[s] = self.d
        for s in right_qubits:
            estr += f",a{s}"
            self.size_dict[s] = self.d
        
        ind = 2*N + 1
        tn_cnt = 2*N

        for _ in range(self.layers):
            # add left layer
            for i in range(N-1):
                lt, lb = left_qubits[i], left_qubits[i+1]
                rt, rb = oe.get_symbol(ind), oe.get_symbol(ind+1)
                estr += f",{lt}{lb}{rt}{rb}"
                ind += 2

                left_qubits[i] = rt 
                left_qubits[i+1] = rb
                self.size_dict[rt] = self.chi
                self.size_dict[rb] = self.chi

                self.left_mps_inds.append(tn_cnt)
                tn_cnt += 1
            
            # add right layer
            for i in range(N-1):
                lt, lb = right_qubits[i], right_qubits[i+1]
                rt, rb = oe.get_symbol(ind), oe.get_symbol(ind+1)
                estr += f",{lt}{lb}{rt}{rb}"
                ind += 2

                right_qubits[i] = rt 
                right_qubits[i+1] = rb
                self.size_dict[rt] = self.chi
                self.size_dict[rb] = self.chi

                self.right_mps_inds.append(tn_cnt)
                tn_cnt += 1
        print(self.size_dict)


        self.measurement_inds = []
        # measurement layer
        for i in range(N-1):
            ql, qr = left_qubits[i], right_qubits[i]
            self.size_dict[qr] = self.chi
            estr += f",{ql}{qr}"
            self.measurement_inds.append(tn_cnt)
            tn_cnt += 1

        #  hamiltonian layer
        ql, qr = left_qubits[-1], right_qubits[-1]
        ll = oe.get_symbol(ind)
        estr += f",{ql}{qr}{ll}"
        self.label_inds = [tn_cnt]
        self.size_dict[ll] = self.l
        ind += 1
        tn_cnt += 1

        
        self.estr_mps = estr[1:]
        self.tn_cnt = tn_cnt
            
            

        # Split the einsum with comma
        einsum_str_split = self.estr_mps.split(",")
        counts = [len(einsum_str_split[i]) for i in range(len(einsum_str_split))]


        self.estr_mps = self.estr_mps + "->a{}".format(ll)

        self.n_tensors = len(counts)
        print(self.size_dict)
        self.ops = [torch.empty(1)] * self.tn_cnt
        self._partial_trace(0)
        self._set_ops()
        self.set_path_mps()

    def forward(self, X: torch.Tensor, normalize_state: bool = False, no_scaling: bool = False):
        X = X.to(self.device).to(self.dtype)
        if normalize_state:
            X = X / torch.norm(X, dim=-1).unsqueeze(-1)

        self._set_ops(not no_scaling)

        for l in range(self.layers):
            for i in range(self.N):
                self.ops[self.left_qubits_inds[l*self.N + i]] = X[i]
                self.ops[self.right_qubits_inds[l*self.N + i]] = X[i]
        
        # measurement qubits
        return oe.contract(self.estr_mps, *self.ops, backend="torch", optimize=self.path) 
    
    def _set_ops(self, scaling: bool = False):

        for i, ind in enumerate(self.left_mps_inds): 
            self.ops[ind] = self.umps.params[i] * (self.scale if scaling else 1)
        for i, ind in enumerate(self.right_mps_inds):
            self.ops[ind] = self.umps.params[i].conj() * (self.scale if scaling else 1)
        self.ops[self.label_inds[0]] = oe.contract("ab,bcl,dc->adl", *[self.r_tensor, self.projector, self.r_tensor.conj()])
        
        # label measure
        # label measure
        # self.ops[self.label_inds[0]] = self.hamiltonian
        # self.ops[self.label_inds[1]] = self.hamiltonian

        # project_0 = torch.tensor([[1, 0], [0, 0]], dtype=self.dtype, device=self.device)
        # project_1 = torch.tensor([[0, 0], [0, 1]], dtype=self.dtype, device=self.device)
        # self.ops[self.label_inds[2]] = torch.stack([project_0, project_1], dim=-1)
    
    def _partial_trace(self, n : int):
        # partial trace
        for i, ind in enumerate(self.measurement_inds):
            if i >= n:
                self.ops[ind] = self.projection 
            else:
                self.ops[ind] = torch.eye(self.chi, dtype=self.dtype, device=self.device)

    def _set_canonical_mps(self, smps: SimpleMPS):

        assert smps.N == self.N
        assert smps.d == self.d
        assert smps.chi == self.chi
        # self.smps = copy.deepcopy(smps)
        params = smps.mps.convert_to_canonical()

        # smps.mps.convert_to_canonical()
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
        self.umps.set_unitaries(MPS_list)
        self.umps.check_unitary()
        self.r_tensor.data[:] = r
        self._set_ops()
    
    # def _diagonalize_hamiltonian(self):
    #     h = self.ham_tensor @ self.Sz @ self.ham_tensor.conj().T
    #     e, V = torch.linalg.eigh(h)
    #     self.ham_tensor.data[:] = torch.diag(e)
    
    
    def _normalize_with_output(self, output: torch.Tensor):
        with torch.no_grad():
            l1 = torch.abs(output).mean() 
            n_mps = len(self.left_mps_inds)  + len(self.right_mps_inds)
            scale = math.exp((-math.log(l1)) / n_mps)
            self.scale = scale * self.scale
        # self._initialize_ops(scale)

    def set_unitaries(self, unitaries: list[torch.Tensor]):
        self.umps.set_unitaries(unitaries)

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

    def _get_path(self,einsum_str: str, optimize: str):
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
        self.path, self.path_info = self._get_path(self.estr_mps, optimize)
        print("Found the path")

    def embed_data(self, data: torch.Tensor, method: str):
        assert data.shape[1] == self.N, "Data must have the same number of columns as the number of qubits"
        if method == "cos":
            return self.embed_data_cos(data)
        elif method == "simple":
            return self.embed_data_simple(data)

    @staticmethod
    def embed_data_simple(X: torch.Tensor):
        """
        X is assumed to be a 1D tensor of shape (N,)
        Note this function can also handle batched data
        """
        # X  = torch.stack([torch.cos(X * np.pi / 2), torch.sin(X * np.pi / 2)], dim=-1)
        # X = torch.stack([X, 1-X], dim=-1)
        X = torch.stack([torch.cos(X * np.pi / 2), torch.sin(X * np.pi / 2)], dim=-1)
        X = X / torch.norm(X, dim=-1).unsqueeze(-1)
        return X



    @staticmethod
    def embed_data_cos(data: torch.Tensor):

        batch_size = data.shape[0]
        N = data.shape[1]
        assert data.ndim == 2, "Data must be a 2D tensor of shape (batch_size, N)"

        X = data.reshape(-1, 1, 1)  # Reshape x to (batch_size * N, 1, 1) for broadcasting

        # Define the Pauli-Y matrix and expand its dimensions
        py = torch.tensor([[0, 1/2], [-1/2, 0]], dtype=data.dtype).unsqueeze(0)

        # Compute the matrix exponential for each element in parallel
        Ry_batch = torch.linalg.matrix_exp(py * np.pi * X)

        return Ry_batch.reshape(batch_size,N, 2, 2)



from torch import nn
import torch
import numpy as np
from scipy.stats import unitary_group, ortho_group
import tensornetwork as tn


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

        self.initialize_params()
    
    def initialize_params(self, std = 1e-3):

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
        for i, param in enumerate(params):
            self.params[i].data[:] = param