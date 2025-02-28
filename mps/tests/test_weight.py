import pytest
import torch
from mps.umps import uMPS
from mps.tpcp_mps import MPSTPCP
import numpy as np
from scipy.stats import unitary_group, ortho_group

class Test_UMPS_weightedMPS:
    def setup_method(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.N = 16
        self.chi = 2
        self.d = 2
        self.layers = 1

        # Generate random input data
        self.rand_input = torch.randn(100, self.N, 2, dtype=torch.float64)
        self.rand_input /= torch.norm(self.rand_input, dim=-1, keepdim=True)
        self.random_Us = torch.stack([self.random_unitary(dtype=torch.float64, device=torch.device("cpu")) for _ in range(self.N - 1)])


    def test_rand_input_normalized(self):
        norms = torch.norm(self.rand_input, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms)), "rand_input is not normalized"


    def random_unitary(self, dtype=torch.float64, device=torch.device("cpu")):
        """
        Generate a random unitary tensor based on the dtype.

        Args:
            dtype (torch.dtype): The desired data type of the tensor.
            device (torch.device): The device on which to create the tensor.

        Returns:
            torch.Tensor: A randomly initialized unitary tensor of shape (chi^2, chi^2).
        """
        if dtype == torch.float64:
            # Generate a random orthogonal matrix for real-valued tensors
            unitary = torch.from_numpy(ortho_group.rvs(self.chi ** 2)).to(dtype=dtype, device=device)
        elif dtype == torch.complex128:
            # Generate a random unitary matrix for complex-valued tensors
            unitary = torch.from_numpy(unitary_group.rvs(self.chi ** 2)).to(dtype=dtype, device=device)
        else:
            raise ValueError("Unsupported dtype for random_unitary. Use torch.float64 or torch.complex128.")
        
        return unitary.reshape(self.chi**2, self.chi**2)

    def test_random_unitary_random_input(self):
        """
        Test the output match for random binary input and a random unitary.

        The input is random binary (0 or 1) and the unitary is slightly off the singlet-triplet unitary.
        This test ensures that the output of the MPSTPCP model matches the output of the uMPS model for this specific case.
        """

        N = 2
        U = self.random_unitary(dtype=torch.float64, device=torch.device("cpu"))

        # Generate random binary input where either 0-th or 1-th element is 1
        rand_input = torch.zeros((100, N, 2), dtype=torch.float64)  # Initialize with zeros
        one_inds = torch.randint(0, 2, (100, N))  # Vector of 0 or 1
        rand_input[torch.arange(100).unsqueeze(1), torch.arange(N), one_inds] = 1  # Set either 0-th or 1-th element to 1

        rand_input /= torch.norm(rand_input, dim=-1, keepdim=True)
        tpcpmps_model = MPSTPCP(N=N, K=1, d=self.d, with_identity=False)
        tpcpmps_model.kraus_ops.init_params(init_with=U.reshape(1, 1, 4, 4))
        tpcpmps_output = tpcpmps_model(rand_input, normalize=False)

        mps_model = uMPS(N=N, chi=self.chi, d=self.d, l=self.d, layers=self.layers, device=torch.device("cpu"), init_with_identity=False)
        mps_model.initialize_MPS(init_with=U.reshape(1, 2, 2, 2, 2))
        mps_output = mps_model(rand_input.permute(1, 0, 2), normalize=False)

        assert tpcpmps_output.shape == (100,), "tpcpmps_output is not a vector with 100 elements"
        assert mps_output.shape == (100,), "mps_output is not a vector with 100 elements"

        # Calculate the last output using hard-coded formula
        for i, x in enumerate(rand_input):
            # Get first element of input
            input1 = x[0, :]
            input2 = x[1, :]
            psi = torch.kron(input1, input2)
            assert torch.allclose(torch.norm(psi), torch.tensor(1.0, dtype=torch.float64)), "psi is not normalized"
            psi = U @ psi

            # the probability of the second element being 0
            prob0 = torch.abs(psi[0])**2 + torch.abs(psi[2])**2

            assert torch.allclose(prob0, tpcpmps_output[i]), "The probability of the second element being 0 does not match"
            assert torch.allclose(prob0, mps_output[i]), "The probability of the second element being 0 does not match"

    def test_outputs_match_identity(self):
        umps_model = uMPS(
            N=self.N,
            chi=self.chi,
            d=self.d,
            l=self.d,
            layers=self.layers,
            device=torch.device("cpu"),
            init_with_identity=True
        )
        tpcpmps_model = MPSTPCP(N=self.N, K=1, d=self.d, with_identity=True)

        # Get outputs from both models
        tpcpmps_output = tpcpmps_model(self.rand_input)
        umps_output = umps_model(self.rand_input.permute(1, 0, 2))

        # Assert that the outputs are close
        assert torch.allclose(tpcpmps_output, umps_output, atol=1e-6), "Outputs from uMPS and MPSTPCP do not match"
    


    def test_outputs_match_random_unitary(self):

        # # Initialize uMPS with the random unitary
        umps_model = uMPS(
            N=self.N,
            chi=self.chi,
            d=self.d,
            l=self.d,
            layers=self.layers,
            device=torch.device("cpu"),
            init_with_identity=False
        )

        umps_model.initialize_MPS(init_with=self.random_Us.reshape(-1, self.chi, self.chi, self.chi, self.chi))
        tpcpmps_model = MPSTPCP(N=self.N, K=1, d=self.d, with_identity=False)
        tpcpmps_model.kraus_ops.init_params(init_with=self.random_Us.reshape(-1, 1, self.chi**2, self.chi**2))  # Adjust based on actual method

        # # Get outputs from both models
        tpcpmps_output = tpcpmps_model(self.rand_input)
        umps_output = umps_model(self.rand_input.permute(1, 0, 2))

        # # Assert that the outputs are close
        assert torch.allclose(tpcpmps_output, umps_output, atol=1e-6), "Outputs from uMPS and MPSTPCP with random unitary do not match"
