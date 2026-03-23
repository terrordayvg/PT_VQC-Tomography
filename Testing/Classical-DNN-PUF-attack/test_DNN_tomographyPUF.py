import pytest
import numpy as np
import sys, os

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Classical-DNN-PUF-attack")))

import DNN_tomographyPUF


# ================================================================
# MODEL TEST
# ================================================================

def test_build_model_compiles():
    model = DNN_tomographyPUF.build_model(input_dim=2**2, output_dim=3)

    assert model.loss == "mse"
    assert hasattr(model.optimizer, "learning_rate")

    x = np.random.rand(5, 2**2)
    y = model.predict(x)

    assert y.shape == (5, 3)


# ================================================================
# DATASET TEST
# ================================================================

def test_generate_dataset():
    N = 10
    dim_ancilla = 2
    dim_target = 2
    shots = 1024   

    X1, Y1 = DNN_tomographyPUF.generate_dataset(
        N, dim_ancilla, dim_target, shots, base_seed=0
    )

    X2, Y2 = DNN_tomographyPUF.generate_dataset(
        N, dim_ancilla, dim_target, shots, base_seed=100
    )

    assert not np.allclose(X1, X2)
    assert not np.allclose(Y1, Y2)

    assert X1.shape == (N, 2**dim_ancilla)
    assert Y1.shape == (N, dim_target)

# ================================================================
# CIRCUIT TESTS
# ================================================================

def test_initial_ry_rotations():
    dim_ancilla = 2
    dim_target = 3

    angles = np.array([0.1, 0.2, 0.3])
    unitary = np.eye(2**dim_target)

    qc = DNN_tomographyPUF.build_qpe_puf_circuit(
        dim_ancilla, dim_target, angles, unitary
    )

    ops = qc.data

    ry_angles = [
        instr.params[0]
        for instr, _, _ in ops
        if instr.name == "ry"
    ]

    assert len(ry_angles) == dim_target
    assert np.allclose(ry_angles, angles)


def test_controlled_unitary_repetitions():
    dim_ancilla = 3
    dim_target = 1

    angles = np.array([0.1])
    unitary = np.eye(2)

    qc = DNN_tomographyPUF.build_qpe_puf_circuit(
        dim_ancilla, dim_target, angles, unitary
    )

    ops = qc.data

    cu_ops = [
        instr for instr, _, _ in ops
        if ("cu" in instr.name.lower() or "unitary" in instr.name.lower())
    ]

    expected_reps = sum(2**j for j in range(dim_ancilla))

    assert len(cu_ops) == expected_reps


# ================================================================
# PANDAS INTEGRATION TEST 
# ================================================================

def test_dataframe_conversion():
    X = np.random.rand(5, 4)
    Y = np.random.rand(5, 3)

    df = DNN_tomographyPUF.dataset_to_dataframe(X, Y)

    # Check shape
    assert df.shape == (5, 7)

    # Check columns exist
    assert "p_0" in df.columns
    assert "angle_0" in df.columns


# ================================================================
# CSV SAVE TEST
# ================================================================

def test_save_csv(tmp_path):
    import pandas as pd

    X = np.random.rand(5, 4)
    Y = np.random.rand(5, 3)

    df = DNN_tomographyPUF.dataset_to_dataframe(X, Y)

    file_path = tmp_path / "test.csv"

    DNN_tomographyPUF.save_csv(df, str(file_path))

    assert file_path.exists()

    loaded = pd.read_csv(file_path)

    assert loaded.shape == df.shape


# ================================================================
# POSTGRES TEST 
# ================================================================

def test_postgres_optional():
    """
    This test will NOT fail if PostgreSQL is not running.
    It only runs if connection is valid.
    """

    try:
        from sqlalchemy import create_engine

        engine = create_engine("postgresql+psycopg2://postgres@localhost:5432/qpuf_db")

        conn = engine.connect()
        conn.close()

    except Exception:
        pytest.skip("PostgreSQL not available, skipping test")
    ops = qc.data

    # Identify controlled-U operations
    cu_ops = [instr for instr, _, _ in ops if "cu" in instr.name or "unitary" in instr.name]

    # Expected number of controlled-U repetitions:
    expected_reps = sum(2**j for j in range(dim_ancilla))

    assert len(cu_ops) == expected_reps
