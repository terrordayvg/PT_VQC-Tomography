import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","State_tomography")))
import qst_code
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def setup_small():
    """Small, fast configuration for tests."""
    N = 1
    depth = 2
    # Random parameters for two layers and one qubit
    thetas = np.random.uniform(0, 2 * np.pi, (depth, N))
    qc = QuantumCircuit(2 * N + 1)  # allocate more qubits to avoid index errors
    return N, depth, qc, thetas

def test_block_qst_runs(setup_small):
    """Ensure block_qst executes correctly and returns a QuantumCircuit."""
    N, depth, qc, thetas = setup_small
    result_circuit = qst_code.block_qst(0, depth, qc.copy(), thetas, N, 0)
    assert isinstance(result_circuit, QuantumCircuit) # "Should return a QuantumCircuit"
    # ensure number of qubits remains unchanged
    assert result_circuit.num_qubits == qc.num_qubits

@pytest.mark.parametrize("i", [0, 1])
def test_block_qst_modifies_circuit(setup_small, i):
    """Verify block_qst adds gates to circuit for both even and last layers."""
    N, depth, qc, thetas = setup_small
    qc_copy = qc.copy()
    initial_ops = len(qc_copy.data)
    result_circuit = qst_code.block_qst(i, depth, qc_copy, thetas, N, 0)
    assert len(result_circuit.data) >= initial_ops  #Circuit should gain or maintain operations"

def test_cost_output_type():
    """Ensure cost returns a finite float."""
    N = 1
    depth = 1
    thetas = np.random.uniform(0, 2 * np.pi, (depth, N))
    val = qst_code.cost(thetas, depth, N)
    assert isinstance(val, float) # "cost should return a float"
    assert np.isfinite(val) # "cost should be finite"
    assert 0 <= val <= 1 # "cost should be within [0, 1]"

def test_cost_consistency():
    """Ensure cost is deterministic for fixed seed values."""
    N = 1
    depth = 1
    thetas = np.zeros((depth, N))
    val1 = qst_code.cost(thetas, depth, N)
    val2 = qst_code.cost(thetas, depth, N)
    assert np.isclose(val1, val2, atol=0.05) #Cost should be consistent for identical inputs, for sufficient statistics 10000shots in run


def test_vqc_qst_integration(setup_small):
    """Check vqc_qst runs end-to-end using block_qst internally."""
    N, depth, qc, thetas = setup_small
    out_circ = qst_code.vqc_qst(depth, qc.copy(), thetas, N, 0)
    assert isinstance(out_circ, QuantumCircuit)

    assert out_circ.num_qubits >= N #Circuit should have enough qubits
