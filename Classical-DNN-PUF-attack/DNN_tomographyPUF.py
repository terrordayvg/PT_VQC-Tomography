"""
Generalized QPUF/QPE + DNN model
---------------------------------------------
"""
# ================================================================
# Imports
# ================================================================

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ================================================================
# QPE Utility Functions
# ================================================================

def _int_to_bitstring(i, width):
    """Convert integer to zero-padded binary string (MSB-left)."""
    return format(i, f'0{width}b')


def qft_dagger(qc: QuantumCircuit, n: int):
    """Inverse QFT (with swaps)."""
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    for j in range(n - 1, -1, -1):
        for k in range(j - 1, -1, -1):
            qc.cp(-np.pi / (2 ** (j - k)), k, j)
        qc.h(j)


# ================================================================
# QPUF/QPE Circuit Builder
# ================================================================

def build_qpe_puf_circuit(dim_ancilla, dim_target, angles, unitary):
    """
    Build a QPE-based PUF circuit:

    - Ancilla (dim_ancilla)
    - Target  (dim_target)
    - RY(angles[i]) initialization on each target qubit
    - Controlled-U repeated exactly 2**j times for each ancilla j
    - Inverse QFT
    - Measurement of ancilla
    """

    n_anc = dim_ancilla
    n_tgt = dim_target

    q_anc = QuantumRegister(n_anc, "anc")
    q_tgt = QuantumRegister(n_tgt, "tgt")
    c_anc = ClassicalRegister(n_anc, "c")
    qc = QuantumCircuit(q_anc, q_tgt, c_anc)

    # Initial target rotations
    for i in range(n_tgt):
        qc.ry(angles[i], q_tgt[i])

    # Hadamards on ancilla
    for i in range(n_anc):
        qc.h(q_anc[i])

    # Build controlled-U gate
    U_gate = UnitaryGate(unitary)
    CU = U_gate.control(1)

    # Append controlled-U 2**j times for ancilla j
    for j in range(n_anc):
        reps = 2 ** j
        qubits = [q_anc[j]] + [q_tgt[k] for k in range(n_tgt)]
        for _ in range(reps):
            qc.append(CU, qubits)

    # Inverse QFT on ancilla
    qft_dagger(qc, n_anc)

    qc.measure(q_anc, c_anc)
    print(qc)

    return qc


# ================================================================
# High-Level Sample Generation
# ================================================================

def generate_puf_sample(seed, dim_ancilla, dim_target, shots=2048, unitary_seed=None, return_counts=False):
    """Generate a single QPUF sample: ancilla probability vector + target angles."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, size=dim_target)

    # Haar unitary
    if unitary_seed is not None:
        U = random_unitary(2 ** dim_target, seed=unitary_seed).data
    else:
        U = random_unitary(2 ** dim_target).data

    qc = build_qpe_puf_circuit(dim_ancilla, dim_target, angles, U)

    sim = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert to probability vector
    prob_vec = np.zeros(2 ** dim_ancilla)
    for i in range(2 ** dim_ancilla):
        key = _int_to_bitstring(i, dim_ancilla)
        prob_vec[i] = counts.get(key, 0) / shots


    if return_counts:
        return prob_vec, angles, counts
    return prob_vec, angles


# ================================================================
# Dataset Generator
# ================================================================

def generate_dataset(N, dim_ancilla, dim_target, shots=2048, base_seed=0, unitary_seed=None):
    """
    Generate dataset (X, Y):

    X: probability vectors (N × 2**dim_ancilla)
    Y: target rotation angles  (N × dim_target)
    """
    X = np.zeros((N, 2 ** dim_ancilla))
    Y = np.zeros((N, dim_target))

    for i in range(N):
        p, a = generate_puf_sample(
            seed=base_seed + i,
            dim_ancilla=dim_ancilla,
            dim_target=dim_target,
            shots=shots,
            unitary_seed=unitary_seed,
            return_counts=False
        )
        X[i] = p
        Y[i] = a


    return X, Y


# ================================================================
# Machine Learning Model
# ================================================================

def build_model(dim_ancilla, dim_target):
    """
    A compact Keras regression model:
       input:  2**dim_ancilla
       output: dim_target  (estimate RY angles)
    """
    input_dim = 2 ** dim_ancilla

    model = Sequential([
        Dense(8, activation="relu", input_shape=(input_dim,)),
        Dense(4, activation="relu"),
        Dense(dim_target)  # regression output
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )
    return model


# ================================================================
# Training Utility
# ================================================================

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=40, batch=32):
    """Fit the neural network."""
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch,
        verbose=1
    )
    return history


# ================================================================
# Example Usage
# ================================================================

if __name__ == "__main__":

    # ---------------- CONFIG ----------------
    dim_ancilla = 2
    dim_target  = 2
    shots       = 4096
    unitary_seed = 1234     

    N_train = 100
    N_test  = 100

    print("Generating training set…")
    X_train, Y_train = generate_dataset(
        N_train, dim_ancilla, dim_target, shots, base_seed=0, unitary_seed=unitary_seed
    )

    print("Generating test set…")
    X_test, Y_test = generate_dataset(
        N_test, dim_ancilla, dim_target, shots, base_seed=10000, unitary_seed=unitary_seed
    )

    print("Building model…")
    model = build_model(dim_ancilla, dim_target)

    print("Training model…")
    train_model(model, X_train, Y_train, X_test, Y_test, epochs=10)

    # Evaluate
    preds = model.predict(X_test)
    mse = np.mean((preds - Y_test) ** 2)
    print("\n==============================")
    print(" Final evaluation MSE:", mse)
    print("==============================")

    # Example prediction
    idx = 5
    print("\nExample test sample:")
    print("True angles :", Y_test[idx])
    print("Pred angles :", preds[idx])
