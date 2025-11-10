import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Process_tomography")))
import New_process_tomo
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_inputs():
    qc = QuantumCircuit(1)
    aux_circuit = QuantumCircuit(1)
    Mu = MagicMock()  # placeholder for unitary gate
    thetas = np.ones(3)  # small array of parameters
    n = 1
    X = np.eye(2)
    depth = 1
    q = MagicMock()
    c = MagicMock()
    CW = MagicMock()
    i = 0
    return qc, aux_circuit, Mu, thetas, n, X, MagicMock(), depth, q, c, CW, i

def test_single_4term_psr_returns_number(dummy_inputs):
    qc, aux_circuit, Mu, thetas, n, X, cont_block, depth, q, c, CW, i = dummy_inputs

    # Mock dependent functions so no real simulation is run
    with patch("New_process_tomo.create_whole_circuit", return_value=MagicMock()), \
         patch("New_process_tomo.sum_frequencies_all", side_effect=[1.0, 0.8, 0.6, 0.4]):
        
        result = New_process_tomo.single_4term_psr(qc, aux_circuit, Mu, thetas, n, X, cont_block, depth, q, c, CW, i)

    # Check that it returns a float and within a reasonable numeric range
    assert isinstance(result, (float, np.floating)), "Result must be a number"
    assert not np.isnan(result), "Result should not be NaN"



def test_sum_frequencies():
    n = 2
    q = QuantumRegister(n + 1, 'q')
    c = ClassicalRegister(n, 'c') 
    circuit = QuantumCircuit(q,c)

    # --- Mock Qiskit components ---
    fake_backend = MagicMock()
    fake_results = MagicMock()
    
    # Simulate measurement results
    fake_counts = {'0': 800, '1': 200}
    
    # Mock behaviors:
    with patch("New_process_tomo.Aer.get_backend", return_value=fake_backend), \
         patch("New_process_tomo.transpile", return_value=circuit), \
         patch("New_process_tomo.marginal_counts") as mock_marginal:
        
        # marginal_counts(...).get_counts()['0'] should return 800
        mock_marginal.return_value.get_counts.return_value = fake_counts
        
        # Mock backend.run().result()
        fake_backend.run.return_value.result.return_value = fake_results
        
        result = New_process_tomo.sum_frequencies(circuit, n)

    # Check results
    assert isinstance(result, float)
    assert 0 <= result <= 1, "Frequency must be between 0 and 1"

    # Ensure mocks were called
    fake_backend.run.assert_called()
    mock_marginal.assert_called()



def test_cohe_sum_2():
    n = 2
    q = QuantumRegister(n + 1, 'q')
    circuit = QuantumCircuit(q)

    cont_block = MagicMock(return_value=circuit)  # mock to avoid real gate logic
    thetas = np.ones((2, 3 * n))
    depth = 1

    result = New_process_tomo.cohe_sum_2(circuit, n, q, cont_block, thetas, depth)

    #Should return a QuantumCircuit
    assert isinstance(result, QuantumCircuit)

    #cont_block should have been called at least once
    assert cont_block.call_count > 0, "Expected cont_block to be called"

    #Should contain gates (X, CRZ, CRY, H)
    gate_names = [instr[0].name for instr in result.data]
    for expected_gate in ["x", "crz", "cry", "h"]:
        assert any(g == expected_gate for g in gate_names), f"Expected {expected_gate} gate in circuit"



def test_cont_block():
    n = 2
    q = QuantumRegister(2 * n, 'q')  # must have enough qubits for control + targets
    circuit = QuantumCircuit(q)

    # thetas[s] should be an array with 3*n elements
    thetas = np.ones((1, 3 * n))  # shape (s_count, parameters_per_s)
    s = 0  # layer index
    z = 0  # control index

    result = New_process_tomo.cont_block(s, z, n, circuit, q, thetas)

    # Should return a QuantumCircuit
    assert isinstance(result, QuantumCircuit)

    # Should contain controlled rotations
    gate_names = [instr.operation.name for instr in result.data]
    assert any(name in gate_names for name in ["crz", "cry"]), "Expected CRZ or CRY gates in circuit"

    # No errors and non-empty circuit
    assert len(gate_names) > 0, "Expected some gates to be added"