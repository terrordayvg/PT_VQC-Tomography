import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate
from qiskit.quantum_info import random_unitary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","State_tomography")))
import attack_qst_code
from unittest.mock import patch, MagicMock



@patch("attack_qst_code.QPUF", return_value=MagicMock())
@patch("attack_qst_code.swap", return_value=(MagicMock(), 1, 2))
@patch("attack_qst_code.gate_vqc_qst", return_value=MagicMock())
def test_gradient_output_shape(mock_gate, mock_swap, mock_qpuf):
    """Check that gradient() returns array of correct shape and values."""
    depth, A, T = 2, 2, 1
    X = MagicMock()
    thetas = np.zeros((2 ** A, depth, T))

    grad = attack_qst_code.gradient(depth, A, thetas, T, X)

    # Expected shape: (2**A, depth, T)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2 ** A, depth, T)

    # Should contain finite numerical values
    assert np.all(np.isfinite(grad))


@patch("attack_qst_code.QPUF", return_value=MagicMock())
@patch("attack_qst_code.swap", return_value=(MagicMock(), 1, 2))
@patch("attack_qst_code.gate_vqc_qst", return_value=MagicMock())
def test_gradient_small_values(mock_gate, mock_swap, mock_qpuf):
    """Ensure gradients are within reasonable range (since mock returns fixed freq)."""
    depth, A, T = 1, 1, 1
    X = MagicMock()
    thetas = np.random.rand(2 ** A, depth, T)

    grad = attack_qst_code.gradient(depth, A, thetas, T, X)

    # Should produce small numeric differences
    assert np.all(np.abs(grad) <= 1.0)


@patch("attack_qst_code.QPUF", side_effect=Exception("Quantum failure"))
def test_gradient_raises_on_qpuf_error(mock_qpuf):
    """If QPUF fails, gradient() should raise an exception."""
    depth, A, T = 1, 1, 1
    X = MagicMock()
    thetas = np.zeros((2 ** A, depth, T))

    with pytest.raises(Exception):
        attack_qst_code.gradient(depth, A, thetas, T, X)

def test_QPUF_builds_expected_circuit_structure():
    """Ensure QPUF builds a valid QuantumCircuit with expected structure."""
    # Parameters
    T = 1
    A = 2
    dq = 2 * T + A + 2
    dc = A + 1 + 2 ** A

    # Create dummy registers and circuit
    qreg = QuantumRegister(dq, 'quantum a')
    creg = ClassicalRegister(dc, 'classical a')
    circuit = QuantumCircuit(qreg, creg)

    # Create a random unitary gate as input
    U = random_unitary(2 ** T)
    X = UnitaryGate(U)

    # Call the QPUF builder
    out_circuit = attack_qst_code.QPUF(X, circuit, T, A, creg)

    # Assertions on type and structure
    assert isinstance(out_circuit, QuantumCircuit)
    # Expect at least some gates and measurements added
    assert len(out_circuit.data) > 0
    # Ensure there are measurement operations
    measure_ops = [op for op, _, _ in out_circuit.data if op.name == "measure"]
    assert len(measure_ops) > 0, "QPUF should include measurements"

    # Check that inverse QFT gates (qfti) appear
    qfti_gates = [op for op, _, _ in out_circuit.data if "qfti" in op.name]
    assert len(qfti_gates) > 0, "Inverse QFT gate should be included"

    # Ensure the number of classical bits matches expectation
    assert out_circuit.num_clbits == dc

    assert out_circuit.num_qubits == dq
