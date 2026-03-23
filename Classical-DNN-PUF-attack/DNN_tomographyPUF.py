"""
Full QPUF/QPE + DNN Pipeline
---------------------------------------------
Includes:
- Dataset generation
- Pandas integration
- PostgreSQL storage (optional)
- Experiment logging
"""

# ================================================================
# Imports
# ================================================================

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library.generalized_gates import UnitaryGate

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Optional DB
from sqlalchemy import create_engine


# ================================================================
# CONFIG
# ================================================================

USE_POSTGRES = False  # Toggle Postgresql usage (note: you need Psql downloaded for this)
# This will create a database: qpuf_db, with user: postgres and password: password (change this)
# Your PostgresSql table will be called qpuf_train
DB_URI = "postgresql+psycopg2://postgres:password@localhost:5432/qpuf_db"


# ================================================================
# Utilities
# ================================================================

def _int_to_bitstring(i, width):
    return format(i, f'0{width}b')


def get_engine():
    return create_engine(DB_URI)


# ================================================================
# QFT
# ================================================================

def qft_dagger(qc, n):
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)

    for j in range(n - 1, -1, -1):
        for k in range(j - 1, -1, -1):
            qc.cp(-np.pi / (2 ** (j - k)), k, j)
        qc.h(j)


# ================================================================
# Circuit Builder
# ================================================================

def build_qpe_puf_circuit(dim_ancilla, dim_target, angles, unitary):

    q_anc = QuantumRegister(dim_ancilla, "anc")
    q_tgt = QuantumRegister(dim_target, "tgt")
    c_anc = ClassicalRegister(dim_ancilla, "c")

    qc = QuantumCircuit(q_anc, q_tgt, c_anc)

    # Target init
    for i in range(dim_target):
        qc.ry(angles[i], q_tgt[i])

    # Ancilla superposition
    for i in range(dim_ancilla):
        qc.h(q_anc[i])

    # Controlled-U
    U_gate = UnitaryGate(unitary)
    CU = U_gate.control(1)

    for j in range(dim_ancilla):
        reps = 2 ** j
        qubits = [q_anc[j]] + [q_tgt[k] for k in range(dim_target)]

        for _ in range(reps):
            qc.append(CU, qubits)

    qft_dagger(qc, dim_ancilla)

    qc.measure(q_anc, c_anc)

    return qc


# ================================================================
# Sample Generator
# ================================================================

def generate_puf_sample(seed, dim_ancilla, dim_target, shots, unitary_seed=None):

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2*np.pi, size=dim_target)

    if unitary_seed is not None:
        U = random_unitary(2 ** dim_target, seed=unitary_seed).data
    else:
        U = random_unitary(2 ** dim_target).data

    qc = build_qpe_puf_circuit(dim_ancilla, dim_target, angles, U)

    sim = Aer.get_backend("aer_simulator")
    print(qc)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    prob_vec = np.zeros(2 ** dim_ancilla)

    for i in range(2 ** dim_ancilla):
        key = _int_to_bitstring(i, dim_ancilla)
        prob_vec[i] = counts.get(key, 0) / shots

    return prob_vec, angles


# ================================================================
# Dataset Generator
# ================================================================

def generate_dataset(N, dim_ancilla, dim_target, shots, base_seed=0, unitary_seed=None):

    X = []
    Y = []

    for i in range(N):
        p, a = generate_puf_sample(
            seed=base_seed + i,
            dim_ancilla=dim_ancilla,
            dim_target=dim_target,
            shots=shots,
            unitary_seed=unitary_seed
        )

        X.append(p)
        Y.append(a)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


# ================================================================
# Pandas Helpers
# ================================================================

def dataset_to_dataframe(X, Y):

    df_X = pd.DataFrame(X, columns=[f"p_{i}" for i in range(X.shape[1])])
    df_Y = pd.DataFrame(Y, columns=[f"angle_{i}" for i in range(Y.shape[1])])

    return pd.concat([df_X, df_Y], axis=1)


def save_csv(df, filename):
    df.to_csv(filename, index=False)


# ================================================================
# PostgreSQL Helpers
# ================================================================

def save_to_postgres(df, table_name):
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)


def load_from_postgres(table_name):
    engine = get_engine()
    df = pd.read_sql(table_name, engine)
    return df


# ================================================================
# Model
# ================================================================

def build_model(input_dim, output_dim):

    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(output_dim)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return model


# ================================================================
# Training
# ================================================================

def train_model(model, X_train, Y_train, X_val, Y_val):

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    return history


# ================================================================
# Experiment Logging
# ================================================================

def log_experiment(mse, config):

    log_df = pd.DataFrame([{
        "mse": mse,
        "dim_ancilla": config["dim_ancilla"],
        "dim_target": config["dim_target"],
        "shots": config["shots"],
        "train_size": config["train_size"]
    }])

    if USE_POSTGRES:
        engine = get_engine()
        log_df.to_sql("experiments", engine, if_exists="append", index=False)
    else:
        log_df.to_csv("experiments_log.csv", mode="a", header=False, index=False)


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # CONFIG
    dim_ancilla = 2
    dim_target = 2
    shots = 4096
    unitary_seed = 1234

    N_train = 100
    N_test = 100

    # DATA
    print("Generating datasets...")
    X_train, Y_train = generate_dataset(N_train, dim_ancilla, dim_target, shots, 0, unitary_seed)
    X_test, Y_test = generate_dataset(N_test, dim_ancilla, dim_target, shots, 10000, unitary_seed)

    # Convert to DataFrame - pandas
    train_df = dataset_to_dataframe(X_train, Y_train)

    # Save locally
    save_csv(train_df, "train_dataset.csv")

    # Optional DB save
    if USE_POSTGRES:
        save_to_postgres(train_df, "qpuf_train")

    # MODEL
    model = build_model(2 ** dim_ancilla, dim_target)

    print("Training model...")
    train_model(model, X_train, Y_train, X_test, Y_test)

    # EVALUATION
    preds = model.predict(X_test)
    mse = np.mean((preds - Y_test) ** 2)

    print("\nFinal MSE:", mse)

    # LOGGING
    config = {
        "dim_ancilla": dim_ancilla,
        "dim_target": dim_target,
        "shots": shots,
        "train_size": N_train
    }

    log_experiment(mse, config)

    # SAMPLE OUTPUT
    idx = 0
    print("\nExample:")
    print("True :", Y_test[idx])
    print("Pred :", preds[idx])


