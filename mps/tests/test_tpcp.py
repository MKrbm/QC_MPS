import pytest
import torch
import numpy as np
import geoopt
from mps.tpcp_mps import MPSTPCP

class TestMPSTPCP_Basic:
    def setup_method(self):
        # Fix seeds for reproducibility.
        torch.manual_seed(42)
        np.random.seed(42)
        # For MPSTPCP, N is the number of qubits, d is the local dimension,
        # and the Kraus operators act on a space of dimension act_size = d^n with n=2.
        self.N = 2      # Number of qubits (minimal test case)
        self.K = 3      # Number of Kraus operators per layer.
        self.d = 2      # Local qubit dimension.
        self.n = 2      # Number of qubits per Kraus operator.
        self.act_size = self.d ** self.n  # 2^2 = 4
        # Create a default model instance (with with_identity=False so that we can test init_with separately)
        self.model = MPSTPCP(N=self.N, K=self.K, d=self.d, with_identity=False)

    def test_wrong_input_shape(self):
        # The forward() method asserts that input X must have shape (batch_size, N, 2).
        wrong_shape_input = torch.randn(10, self.N - 1, 2, dtype=torch.float64)
        with pytest.raises(AssertionError):
            _ = self.model(wrong_shape_input)

    def test_init_with_wrong_length(self):
        # For MPSTPCP with N=2, self.L = N-1 = 1.
        # Here, we pass a tensor of length 2 instead of 1.
        K = 1
        model = MPSTPCP(N=self.N, K=K, d=self.d, with_identity=False)
        eye_matrix = torch.eye(self.act_size, dtype=torch.float64).reshape(K, self.act_size, self.act_size)
        init_tensor = torch.stack([eye_matrix] * 2)  # Incorrect length: 2 instead of 1
        
        with pytest.raises(ValueError):
            self.model.kraus_ops.init_params(init_with=init_tensor)

    def test_init_with_wrong_matrix_dimension(self):
        # Supply a tensor with incompatible shape (e.g. (1, 3, 3) instead of (1, 4, 4)).
        K = 1
        model = MPSTPCP(N=self.N, K=K, d=self.d, with_identity=False)
        wrong_tensor = torch.eye(3, dtype=torch.float64).reshape(K, 3, 3)
        init_tensor = torch.stack([wrong_tensor])
        with pytest.raises(RuntimeError):
            self.model.kraus_ops.init_params(init_with=init_tensor)

    def test_init_with_non_stiefel(self):
        # Supply an init_with tensor that is not on the Stiefel manifold.
        non_stiefel = torch.ones(self.K, self.act_size, self.act_size, dtype=torch.float64)
        init_tensor = torch.stack([non_stiefel])
        with pytest.raises(ValueError) as excinfo:
            self.model.kraus_ops.init_params(init_with=init_tensor)
        assert "does not represent a point on the Stiefel manifold" in str(excinfo.value)

    def test_dtype_consistency(self):
        # Create a valid initialization using geoopt's random method.
        st = geoopt.Stiefel()
        valid_matrix = st.random((self.K * self.act_size, self.act_size), dtype=torch.float64)
        valid_tensor = valid_matrix.reshape(self.K, self.act_size, self.act_size)
        init_tensor = torch.stack([valid_tensor])
        self.model.kraus_ops.init_params(init_with=init_tensor)
        for param in self.model.kraus_ops:
            assert param.dtype == torch.float64, "Parameter dtype is not torch.float64"

    def test_output_with_identity(self):
        # When with_identity=True, each Kraus operator is set to (1/sqrt(K))*I.
        model_identity = MPSTPCP(N=self.N, K=self.K, d=self.d, with_identity=True)
        batch_size = 10
        input_state = torch.zeros(batch_size, self.N, 2, dtype=torch.float64)
        input_state[:, :, 0] = 1.0
        output = model_identity(input_state, normalize=False)
        expected = torch.ones(batch_size, dtype=torch.float64)
        assert torch.allclose(output, expected, atol=1e-6), "Output with identity initialization is not as expected."

    def test_partial_invalid_site(self):
        batch_size = 1
        dummy_rho = torch.eye(self.act_size, dtype=torch.float64).unsqueeze(0)
        with pytest.raises(ValueError):
            _ = self.model.partial(dummy_rho, site=2)

    def test_normalization_of_input(self):
        batch_size = 5
        X = torch.randn(batch_size, self.N, 2, dtype=torch.float64)
        X_normalized = X / torch.norm(X, dim=-1, keepdim=True)
        output1 = self.model(X, normalize=True)
        output2 = self.model(X_normalized, normalize=False)
        assert torch.allclose(output1, output2, atol=1e-6), "Normalization in forward does not work as expected."

    def test_forward_output_shape(self):
        batch_size = 7
        X = torch.randn(batch_size, self.N, 2, dtype=torch.float64)
        X = X / torch.norm(X, dim=-1, keepdim=True)
        output = self.model(X)
        assert output.shape == (batch_size,), "Forward output shape is incorrect."

    def test_forward_probability_range(self):
        model_identity = MPSTPCP(N=self.N, K=self.K, d=self.d, with_identity=True)
        batch_size = 10
        X = torch.randn(batch_size, self.N, 2, dtype=torch.float64)
        X = X / torch.norm(X, dim=-1, keepdim=True)
        output = model_identity(X)
        assert torch.all((output >= 0) & (output <= 1)), "Output probabilities are not in the range [0, 1]."

    def test_is_tpcp_on_identity(self):
        identity = torch.eye(self.act_size, dtype=torch.float64)
        kraus_identity = (1.0 / np.sqrt(self.K)) * identity
        tensor_identity = torch.stack([kraus_identity] * self.K).reshape(self.K, self.act_size, self.act_size)
        stiefel = geoopt.Stiefel()
        candidate = tensor_identity.reshape(self.K * self.act_size, self.act_size)
        # Using geoopt's check function (or .belongs, depending on your geoopt version)
        assert stiefel.check_point_on_manifold(candidate), "Identity does not belong to the Stiefel manifold."
        assert self.model.kraus_ops.is_tpcp(tensor_identity), "is_tpcp did not return True for identity initialization."
    # === Expected Output Tests ===

    def test_expected_output_trivial_identity(self):
        """
        Case 1:
        When the Kraus operators are identity and the inputs are trivial (each qubit in state [1, 0]),
        the channel acts as the identity. For input |00>, the reduced density matrix for qubit 2 is |0><0|
        so the probability of measuring 0 is 1.
        """
        batch_size = 5
        K = 1
        trivial_input = torch.zeros(batch_size, self.N, 2, dtype=torch.float64)
        trivial_input[:, :, 0] = 1.0
        model_identity = MPSTPCP(N=self.N, K=K, d=self.d, with_identity=True)
        output = model_identity(trivial_input, normalize=False)
        expected = torch.ones(batch_size, dtype=torch.float64)
        assert torch.allclose(output, expected, atol=1e-6), \
            "Trivial identity output does not match expected value of 1 for all samples."

    def test_kraus_operator_shape_for_K_identity(self):
        """
        Case 2:
        When the Kraus operators are set to identity for K > 1, the underlying parameter matrix
        should have shape (K * act_size, act_size), i.e. (K * d^n, d^n).
        For d=2, n=2, and K=2, we expect (2*4, 4) = (8, 4).
        """
        K = 2
        model_identity = MPSTPCP(N=self.N, K=K, d=self.d, with_identity=True)
        expected_shape = (K * self.act_size, self.act_size)
        for param in model_identity.kraus_ops:
            assert param.shape == expected_shape, \
                f"Kraus operator shape is {param.shape}, expected {expected_shape}"

    def test_expected_output_unitary(self):
        """
        Case 3:
        When the Kraus operators are set to a specific unitary matrix and N=2.
        Here we use a permutation unitary defined on the 4-dimensional space.
        For trivial input |00>, the unitary maps the state to another basis vector,
        but the reduced probability (obtained via partial trace) should still be 1.
        """
        # Define a permutation matrix (which is unitary) of shape 4x4.
        U = torch.tensor([[0., 1., 0., 0.],
                          [1., 0., 0., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 1., 0.]], dtype=torch.float64)
        batch_size = 5
        trivial_input = torch.zeros(batch_size, self.N, 2, dtype=torch.float64)
        trivial_input[:, :, 0] = 1.0
        # Expected: for |00>, the state is the first canonical basis vector.
        # Under U, |00> maps to the first column of U, which is [0, 1, 0, 0]^T.
        # Then the reduced density matrix for qubit 2 (after partial trace) yields probability 1.
        init_with = U.reshape(1, 1, self.act_size, self.act_size)
        model_unitary = MPSTPCP(N=self.N, K=1, d=self.d, with_identity=False)
        model_unitary.kraus_ops.init_params(init_with=init_with)
        output = model_unitary(trivial_input, normalize=False)
        expected = torch.zeros(batch_size, dtype=torch.float64)
        assert torch.allclose(output, expected, atol=1e-6), \
            "Expected output for unitary Kraus operator does not match expected value of 1."

    def test_expected_output_random_stiefel(self):
        """
        Case 4:
        When the Kraus operators are randomly initialized on the Stiefel manifold and N=2.
        For a trivial input |00>, we manually compute the channel's output.
        """
        batch_size = 5
        N = 2
        K = 3
        trivial_input = torch.zeros(batch_size, N, 2, dtype=torch.float64)
        trivial_input[:, :, 0] = 1.0
        # Use the default (random Stiefel) initialization.
        model_random = MPSTPCP(N=N, K=K, d=self.d, with_identity=False)
        output_model = model_random(trivial_input, normalize=False)
        # Manually compute the channel action:
        psi = torch.kron(torch.tensor([1.0, 0.0], dtype=torch.float64),
                         torch.tensor([1.0, 0.0], dtype=torch.float64))
        rho_manual = torch.outer(psi, psi)  # 4x4 density matrix for |00><00|

        assert len(list(model_random.kraus_ops.parameters())) == 1

        kraus_param = model_random.kraus_ops[0].reshape(self.K, self.act_size, self.act_size)
        rho_after_list = [E @ rho_manual @ E.conj().t() for E in kraus_param]
        rho_after = sum(rho_after_list)
        # Partial trace over the first qubit.
        rho_reshaped = rho_after.reshape(self.d, self.d, self.d, self.d)
        rho_reduced = torch.einsum("a c a d -> c d", rho_reshaped)
        expected_manual = rho_reduced[0, 0]  # probability of measuring 0 on the second qubit
        # Check that every sample's output matches the manually computed probability.
        for val in output_model:
            assert torch.allclose(val, expected_manual, atol=1e-6), \
                "Random Stiefel output does not match manual computation."

    def test_expected_output_random_stiefel_normalized(self):
        """
        Case 5:
        When the Kraus operators are randomly initialized on the Stiefel manifold and N=2.
        For a normalized random input, we manually compute the channel's output.
        """
        batch_size = 5
        N = 2
        K = 3
        # Generate normalized random input
        random_input = torch.randn(batch_size, N, 2, dtype=torch.float64)
        random_input /= torch.norm(random_input, dim=-1, keepdim=True)
        # Use the default (random Stiefel) initialization.
        model_random = MPSTPCP(N=N, K=K, d=self.d, with_identity=False)
        output_model = model_random(random_input, normalize=False)
        # Manually compute the channel action:
        for i in range(batch_size):
            psi0 = random_input[i, 0, :]
            psi1 = random_input[i, 1, :]
            psi = torch.kron(psi0, psi1)
            rho_manual = torch.outer(psi, psi.conj())  # 4x4 density matrix

            assert len(list(model_random.kraus_ops.parameters())) == 1

            kraus_param = model_random.kraus_ops[0].reshape(K, self.act_size, self.act_size)
            rho_after_list = [E @ rho_manual @ E.conj().t() for E in kraus_param]
            rho_after = sum(rho_after_list)
            # Partial trace over the first qubit.
            rho_reshaped = rho_after.reshape(self.d, self.d, self.d, self.d)
            rho_reduced = torch.einsum("a c a d -> c d", rho_reshaped)
            expected_manual = rho_reduced[0, 0]  # probability of measuring 0 on the second qubit
            assert torch.allclose(output_model[i], expected_manual, atol=1e-6), \
                "Random Stiefel output with normalized input does not match manual computation."
