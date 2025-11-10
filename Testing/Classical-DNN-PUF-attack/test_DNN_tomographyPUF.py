
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Classical-DNN-PUF-attack")))
import DNN_tomographyPUF


def test_build_model_compiles():
    model = DNN_tomographyPUF.build_model(dim_ancilla=2, dim_target=3)

    # Test loss and optimizer
    assert model.loss == "mse"
    assert isinstance(model.optimizer, Adam)

    # Make sure a forward pass works
    import numpy as np
    x = np.random.rand(5, 2 ** 2)  # batch of 5
    y = model.predict(x)
    assert y.shape == (5, 3)


def test_generate_dataset():
    """Different seeds should produce different results (very likely)."""
    N = 10
    dim_ancilla = 2
    dim_target = 2

    X1, Y1 = DNN_tomographyPUF.generate_dataset(N, dim_ancilla, dim_target, base_seed=0)
    X2, Y2 = DNN_tomographyPUF.generate_dataset(N, dim_ancilla, dim_target, base_seed=100)

    # At least one value should differ
    assert not np.allclose(X1, X2)
    assert not np.allclose(Y1, Y2)

#Check if rotations are present in initialization
def test_initial_ry_rotations():
    dim_ancilla = 2
    dim_target = 3
    angles = np.array([0.1, 0.2, 0.3])
    unitary = np.eye(2**dim_target)

    qc = DNN_tomographyPUF.build_qpe_puf_circuit(dim_ancilla, dim_target, angles, unitary)

    # Extract operations
    ops = qc.data

    # Check that each target qubit received an RY gate with the correct angle
    ry_angles = [
        instr.params[0] for instr, qargs, cargs in ops
        if instr.name == "ry"
    ]

    assert len(ry_angles) == dim_target
    assert np.allclose(ry_angles, angles)


#Check that we are doing 2**n control operations on QPE based Unitary.
def test_controlled_unitary_repetitions():
    dim_ancilla = 3
    dim_target = 1
    angles = [0.1]
    unitary = np.eye(2)

    qc = DNN_tomographyPUF.build_qpe_puf_circuit(dim_ancilla, dim_target, angles, unitary)

    ops = qc.data

    # Identify controlled-U operations
    cu_ops = [instr for instr, _, _ in ops if "cu" in instr.name or "unitary" in instr.name]

    # Expected number of controlled-U repetitions:
    expected_reps = sum(2**j for j in range(dim_ancilla))

    assert len(cu_ops) == expected_reps