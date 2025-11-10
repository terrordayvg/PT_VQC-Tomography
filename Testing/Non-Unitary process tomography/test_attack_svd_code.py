import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library import UnitaryGate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Non-Unitary process tomography")))
import attack_svd_code


@pytest.fixture
def mock_inputs():
    """Prepare mock inputs for Forgery_SVD."""
    A = 2  # number of ancilla qubits
    T = 1  # number of target qubits

    # Create a simple 2x2 unitary gate (identity)
    U = UnitaryGate(np.eye(2 ** T, dtype=complex))

    # Mock eigenvalues (just some normalized values)
    eigens = np.array([0.25, 0.75])

    # Mock matrix M as identity gate for simplicity
    M = UnitaryGate(np.eye(2 ** T, dtype=complex))

    return U, T, A, M, eigens


@pytest.mark.parametrize("w", [0, 1, 2])
def test_forgery_svd_runs(mock_inputs, w):
    """Test Forgery_SVD executes without errors and returns valid cyclic deviation."""
    U, T, A, M, eigens = mock_inputs

    deviation = attack_svd_code.Forgery_SVD(U, T, A, w, M, eigens)

    # Check type and range
    assert isinstance(deviation, (int, float)), "Output should be a number."
    assert 0 <= deviation <= 2 ** A / 2, f"Deviation {deviation} out of valid cyclic range."
