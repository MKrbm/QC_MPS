import pytest
import torch
from mps.umps import uMPS
from mps.tpcp_mps import MPSTPCP

class TestUMPSvsTPCPMPS:
    def setup_method(self):
        self.N = 16
        self.chi = 2
        self.d = 2
        self.layers = 1

        # Generate random input data
        self.rand_input = torch.randn(self.N, 100, 2, dtype=torch.float64)
        self.rand_input /= torch.norm(self.rand_input, dim=-1, keepdim=True)

        # Initialize both models

    def test_outputs_match_identity(self):
        umps_model = uMPS(N=self.N, chi=self.chi, d=self.d, l=self.d, layers=self.layers, device=torch.device("cpu"), init_with_identity=True)
        tpcpmps_model = MPSTPCP(N=self.N, K=1, d=self.d, with_identity=True)
        # Get outputs from both models
        tpcpmps_output = tpcpmps_model(self.rand_input.permute(1, 0, 2))
        umps_output = umps_model(self.rand_input)

        # Assert that the outputs are close
        assert torch.allclose(tpcpmps_output, umps_output, atol=1e-6), "Outputs from uMPS and MPSTPCP do not match"

    # def test_outputs_match_random_unitary(self):
    #     umps_model = uMPS(N=self.N, chi=self.chi, d=self.d, l=self.d, layers=self.layers, device=torch.device("cpu"), init_with_identity=True)
    #     tpcpmps_model = MPSTPCP(N=self.N, K=1, d=self.d, with_identity=True)
    #     # Get outputs from both models
    #     tpcpmps_output = tpcpmps_model(self.rand_input.permute(1, 0, 2))
    #     umps_output = umps_model(self.rand_input)

    #     # Assert that the outputs are close
    #     assert torch.allclose(tpcpmps_output, umps_output, atol=1e-6), "Outputs from uMPS and MPSTPCP do not match"

# No need for the __main__ block as pytest will discover the test class automatically