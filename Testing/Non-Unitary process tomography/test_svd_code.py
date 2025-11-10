import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os
from qiskit.circuit.library.generalized_gates import UnitaryGate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Non-Unitary process tomography")))
import svd_code

def test_singu_basic():
    # Smallest system
    N = 1
    depth = 1
    d = 2 ** N

    # Identity unitary → should produce singular values ~1+0i
    U = UnitaryGate(np.eye(d, dtype=complex))

    # Zero parameters for simplicity
    alphas = np.zeros(3 * N * depth)

    # Dummy previous measurement data (must be length d)
    check_real = np.zeros(d)
    check_imag = np.zeros(d)

    # Run function
    singular, pos_real, pos_imag = svd_code.singu(N, U, alphas, check_real, check_imag, depth)

    # Tests
    assert isinstance(singular, np.ndarray)
    assert isinstance(pos_real, np.ndarray)
    assert isinstance(pos_imag, np.ndarray)

    assert singular.shape == (d,)
    assert pos_real.shape == (d,)
    assert pos_imag.shape == (d,)

    # Ensure values are finite real numbers
    assert np.all(np.isfinite(singular.real))
    assert np.all(np.isfinite(singular.imag))
    assert np.all(np.isfinite(pos_real))
    assert np.all(np.isfinite(pos_imag))




def test_loss_svd_basic():
    # Smallest system setup
    N = 1
    depth = 1
    d = 2 ** N

    # Identity target unitary
    U = UnitaryGate(np.eye(d, dtype=complex))

    # Zero angles to keep circuit simple
    alphas = np.zeros(3 * N * depth)

    # Run the loss function
    cost = svd_code.loss_svd(alphas, N, depth, U)

    # Tests
    assert isinstance(cost, float), "loss_svd should return a float"
    assert np.isfinite(cost), "Loss value must be finite"

    assert 0 <= cost <= 1, "Loss should be within logical bounds (overlap distance)"
