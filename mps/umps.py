import opt_einsum as oe
import torch
from torch import nn
import numpy as np
import copy
from pathlib import Path
from scipy.stats import unitary_group, ortho_group

from opt_einsum.testing import build_views
from opt_einsum.contract import PathInfo, ContractExpression


    
class uMPS(nn.Module):
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
    batch_size: int
    use_simple_path: bool
    path: PathInfo | None = None
    patho: ContractExpression | None = None
    
    def __init__(
        self,
        N: int,
        chi: int,
        d: int,
        l: int,
        layers : int,
        batch_size: int = 100,
        init_with_identity: bool = False,
        normalize: bool = True,
        optimize: str = "auto",
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float64,
        init_with: torch.Tensor | None = None
    ):
        super().__init__()
        # super().__init__(*args, **kwargs)
    

        assert d == l == chi, "Data input dimension and class label dimension must be the same"

        self.layers = layers
        self.N = N
        self.chi = chi
        self.d = d
        self.l = l
        self.batch_size = batch_size
        self.optimize = optimize
        self.device = device
        self.dtype = dtype
        self.is_initialized = False
        self.identity = init_with_identity
        self.normalize = normalize

        estr = ""
        left_qubits = [oe.get_symbol(i + 1) for i in range(N)]
        right_qubits = [oe.get_symbol(i + 1) for i in range(N, 2*N)]

        left_qubits_inds = [i + 1 for i in range(N)]
        right_qubits_inds = [i + 1 for i in range(N, 2*N)]

        self.left_mps_inds = []
        self.right_mps_inds = []

        for s in left_qubits:
            estr += f",a{s}"
        
        for s in right_qubits:
            estr += f",a{s}"
        
        ind = 2*N + 1
        tn_cnt = 2*N

        for _ in range(self.layers - 1):
            # add left layer
            for i in range(N-1):
                lt, lb = left_qubits[i], left_qubits[i+1]
                rt, rb = oe.get_symbol(ind), oe.get_symbol(ind+1)
                estr += f",{lt}{lb}{rt}{rb}"
                ind += 2

                left_qubits[i] = rt 
                left_qubits[i+1] = rb

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
                
                self.right_mps_inds.append(tn_cnt)
                tn_cnt += 1


        # Last layers
        label_qubits_inds = [-1, -1]

        ## left layer
        for i in range(N-1):
            lt, lb = left_qubits[i], left_qubits[i+1]
            rt, rb = oe.get_symbol(ind), oe.get_symbol(ind+1)
            label_qubits_inds[0] = ind + 1 ## replace with newest rb index
            estr += f",{lt}{lb}{rt}{rb}"
            ind += 2

            left_qubits[i] = rt 
            left_qubits[i+1] = rb

            self.left_mps_inds.append(tn_cnt)
            tn_cnt += 1

        # right layer
        for i in range(N-1):
            lt, lb = right_qubits[i], right_qubits[i+1]
            # rt, rb = oe.get_symbol(ind), oe.get_symbol(ind+1)
            rt = left_qubits[i]
            rb = oe.get_symbol(ind)
            label_qubits_inds[1] = ind ## replace with newest rb index
            estr += f",{lt}{lb}{rt}{rb}"
            ind += 1
            right_qubits[i] = rt 
            right_qubits[i+1] = rb

            self.right_mps_inds.append(tn_cnt)
            tn_cnt += 1

        # label_qubits = [left_qubits[-1], right_qubits[-1]]
        for i in label_qubits_inds:
            estr += ",a{}".format(oe.get_symbol(i))
            tn_cnt += 1

        
        self.estr = estr[1:]
        self.label_inds = label_qubits_inds
        self.left_inds = left_qubits_inds
        self.right_inds = right_qubits_inds
        
                

        # Split the einsum with comma
        einsum_str_split = self.estr.split(",")
        counts = [len(einsum_str_split[i]) for i in range(len(einsum_str_split))]


        self.estr = self.estr + "->a" 

        self.n_tensors = len(counts)
        self.tensor_shapes = [(d,) * counts[i] for i in range(len(counts))]
        for i, shapes in enumerate(self.tensor_shapes):
            if len(shapes) == 2:
                self.tensor_shapes[i] = (batch_size, d)
        
        self.set_path(batch_size=batch_size, optimize=optimize)
        self.params = nn.ParameterList([nn.Parameter(torch.empty(self.chi, self.chi, self.chi, self.chi)) for _ in range(self.layers * (self.N - 1))])
        self.initialize_MPS(init_with=init_with)
        self._discriminator = self._get_discriminator()
        


    def initialize_MPS(self, init_with: torch.Tensor | None = None):
        """
        Initialize or overwrite the parameters of the MPS tensors.

        Args:
            init_with (torch.Tensor, optional): Tensor containing unitary matrices to initialize MPS.
                                               Must be unitary if provided.
        """
        # if self.is_initialized:
        #     print("MPS is already initialized")
        #     return

        if init_with is not None:
            if init_with.shape != (len(self.params), self.chi, self.chi, self.chi, self.chi):
                raise ValueError(f"init_with tensor must have shape ({len(self.params)}, {self.chi}, {self.chi}, {self.chi}, {self.chi})")
            for i, param in enumerate(self.params):
                u = init_with[i].reshape(self.chi ** 2, self.chi ** 2)
                # Have to match to the unitary evolution of the input. Complex conjugate must be taken.
                u = u.conj().T
                if not self.is_unitary(u):
                    raise ValueError("The provided tensor is not unitary.")
                param.data[:] = u.reshape((self.chi,) * 4).clone()
        else:
            if self.dtype == torch.float64:
                MPS_unitaries = [
                    torch.from_numpy(ortho_group.rvs(self.chi ** 2).reshape((self.chi,) * 4))
                    for _ in range((self.N - 1) * self.layers)
                ]
            elif self.dtype == torch.complex128:
                MPS_unitaries = [
                    torch.from_numpy(unitary_group.rvs(self.chi ** 2).reshape((self.chi,) * 4))
                    for _ in range((self.N - 1) * self.layers)
                ]
            else:
                raise ValueError("Unsupported dtype")

            if self.identity:
                MPS_unitaries = [torch.eye(self.chi ** 2).reshape((self.chi,) * 4) for _ in range((self.N - 1) * self.layers)]

            for i, unitary in enumerate(MPS_unitaries):
                self.params[i] = nn.Parameter(
                    unitary.clone().detach().to(device=self.device, dtype=self.dtype)
                )

        self.is_initialized = True
        print("Initialized MPS params")

    @staticmethod
    def is_unitary(tensor: torch.Tensor, rtol: float = 1e-6) -> bool:
        """
        Check if the given tensor is unitary.

        Args:
            tensor (torch.Tensor): Tensor to check.

        Returns:
            bool: True if the tensor is unitary, False otherwise.
        """
        if tensor.ndim != 2:
            return False
        identity = torch.eye(tensor.size(0), dtype=tensor.dtype, device=tensor.device)
        return torch.allclose(tensor.conj().transpose(-2, -1) @ tensor, identity, rtol=rtol)

    def set_path(self, optimize: str = 'greedy', batch_size: int = 100):
        if self.path is not None:
            print(f"Path is already set, finding the {optimize} path...")
        else:
            print("Path is not set, setting...")
        self.path, self.path_info = self._get_path(self.estr, self.d, optimize, batch_size)
        print("Found the path")

    @staticmethod
    def _get_path(einsum_str: str, d: int, optimize: str, batch_size: int):
        unique_inds = set(einsum_str) - {',', '-', '>'}
        index_size = [d] * len(unique_inds)
        sizes_dict = dict(zip(unique_inds, index_size))
        sizes_dict['a'] = batch_size
        einsum_str = einsum_str.replace("a", "")
        sizes_dict.pop("a")
        einsum_str = einsum_str.replace("->", "")

        views = build_views(einsum_str, sizes_dict)
        path, path_info = oe.contract_path(einsum_str, *views, optimize=optimize)
        return path, path_info

    def _get_discriminator(self):
        ops = [shape for shape in self.tensor_shapes]
        for i, tensor in enumerate(self.params):
            ops[self.left_mps_inds[i]] = tensor
            ops[self.right_mps_inds[i]] = tensor.conj()
        return oe.contract_expression(self.estr, *ops, constants=self.left_mps_inds + self.right_mps_inds, optimize=self.path)

    def forward(self, X: torch.Tensor, label: torch.Tensor | None = None, normalize: bool = True):
        if normalize:
            X = X / torch.norm(X, dim=-1).unsqueeze(-1)

        batch_size = X.shape[1]
        if X.ndim != 3 or X.shape[0] != self.N:
            raise ValueError("Input must be a tensor of shape (N, batch_size, 2)")

        batch = torch.empty(((self.N+1)*2, batch_size, 2), device=self.device, dtype=self.dtype)
        batch[:self.N, :, :] = X
        batch[self.N:-2, :, :] = X
        if label is not None:
            batch[-2:, :, :] = label
        else:
            batch[-2:, :, :] = torch.tensor([1, 0])
        return self._discriminator(*batch, backend="torch")


    def save_to_file(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix != ".pt":
            raise ValueError("Path must end with .pt")
        with torch.no_grad():
            tensors = [unitary.detach().cpu() for unitary in self.params]
            torch.save(tensors, path)
        print(f"Saved MPS unitaries to {path}")

    def load_from_file(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix != ".pt":
            raise ValueError("Path must end with .pt")
        with torch.no_grad():
            tensors = torch.load(path)
            self.set_unitaries(tensors, check_unitary=True)
        print(f"Loaded MPS unitaries from {path}")

    def set_unitaries(self, MPS_unitaries: list[torch.Tensor], check_unitary: bool = True):
        assert len(MPS_unitaries) == self.N - 1, "MPS_unitaries must have the same length as N - 1"
        assert MPS_unitaries[0].shape == (self.chi, self.chi, self.chi, self.chi), "MPS_unitaries must be of shape (chi, chi, chi, chi)"
        assert MPS_unitaries[0].dtype == self.params[0].dtype, "MPS_unitaries must be of the same dtype as the MPS"

        for i in range(len(self.params)):
            self.params[i].data.copy_(MPS_unitaries[i].to(device=self.device))
            if self.params[i].grad is not None:
                self.params[i].grad.zero_()
            if check_unitary:
                u = self.params[i].reshape(self.chi ** 2, self.chi ** 2).detach().cpu().numpy()
                assert np.allclose(u @ u.T.conj(), np.eye(self.chi ** 2)), "Unitary check failed"
        print("Set MPS unitaries")

    def _proj_to_unitary(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Helper function that projects a tensor back onto the unitary manifold using SVD.
        The tensor is first reshaped to a square matrix, then projected via SVD,
        and finally reshaped back to its original shape.
        
        Args:
            tensor (torch.Tensor): The tensor to project.
            
        Returns:
            torch.Tensor: The projected tensor with the same shape as the input.
        """
        orig_shape = tensor.shape
        matrix = tensor.reshape(self.chi**2, self.chi**2)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        return (U @ Vh).reshape(orig_shape)

    # def check_point_on_unitary(self, rtol: float = 1e-5, print_log: bool = False) -> bool:
    #     """
    #     Checks if all MPS parameters are unitary.
        
    #     For each parameter (stored as a 4-index tensor), it reshapes it to a square matrix 
    #     of size (chi**2, chi**2) and confirms that U†U equals the identity matrix.
        
    #     Args:
    #         rtol (float): Relative tolerance to decide near-unitarity.
    #         print_log (bool): If True, prints a message for parameters that are not unitary.
            
    #     Returns:
    #         bool: True if every parameter is unitary, False otherwise.
    #     """
    #     for i, param in enumerate(self.params):
    #         matrix = param.data.reshape(self.chi**2, self.chi**2)
    #         identity = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
    #         if not torch.allclose(matrix.conj().T @ matrix, identity, atol=rtol):
    #             if print_log:
    #                 print(f"Parameter {i} is not unitary.")
    #             return False
    #     return True

    def proj_unitary(self, check_on_unitary: bool = True, print_log: bool = False, rtol: float = 1e-5):
        """
        Projects the MPS parameters onto the unitary group using SVD.
        
        This method iterates over all parameters in self.params and projects any matrix
        (when reshaped to (chi**2, chi**2)) that is not unitary. If check_on_unitary=True,
        the method first checks and projects only those parameters that are not unitary.
        Otherwise, all parameters are re-projected back onto the unitary group.
        
        Args:
            check_on_unitary (bool): If True, only project those parameters that fail the unitarity check.
            print_log (bool): If True, prints debug messages.
            rtol (float): Relative tolerance for checking unitarity.
        """

        for i, param in enumerate(self.params):
            if check_on_unitary and self.is_unitary(param.data, rtol=rtol):
                continue
            else:
                param.data.copy_(self._proj_to_unitary(param.data))
                if print_log:
                    print(f"Parameter {i} projected to unitary.")