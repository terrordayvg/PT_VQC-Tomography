import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HGate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Process_tomography")))
import process_tomography_code



@pytest.fixture(scope="module")
def setup_basic():
	N = 1
	depth = 1
	thetas = np.random.uniform(0, 2*np.pi, depth * N * 3 + 3 * N)
	X = UnitaryGate(process_tomography_code.random_unitary(2**N))
	return N, depth, thetas, X


def test_grad_loss_runs(setup_basic):
	N, depth, thetas, X = setup_basic
	grad = process_tomography_code.grad_loss(thetas, N, X, depth, 0)
	assert isinstance(grad, float) #grad_loss should return a float
	assert np.isfinite(grad) #Gradient value should be finite


def test_process_output_shape():
	N = 1
	depth = 1
	iterations = 1
	cores = 1
	X = UnitaryGate(process_tomography_code.random_unitary(2**N))
	M, qk = process_tomography_code.Process(X, N, iterations, depth, cores)
	assert isinstance(M, np.ndarray) 
	assert M.shape == (2**N, 2**N)
	assert isinstance(qk, list) #qk should be a list of circuits
	assert all(hasattr(c, 'num_qubits') for c in qk) #All elements of qk should be QuantumCircuits
	assert np.iscomplexobj(M) #Output matrix should contain complex numbers

