#Importing libraries
##############################################################
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import assemble, Aer, IBMQ
import matplotlib.pyplot as plt
##############################################################

#Auxilliary functions
##############################################################
def block_qst(i, depth, circuit, thetas, N, k): #Single block of the VQC.
    if i == depth - 1:
        for m in range(N):
            circuit.rx(thetas[i][m], m + k) 
    elif i % 2 == 0:
        for m in range(N):
            circuit.ry(thetas[i][m], m + k)  
        for m in range(N):
            if 2 * m < N - 1:
                    circuit.cnot(2 * m + k  , 2 * m + 1 + k)
    else:
        for m in range(N):
            circuit.rx(thetas[i][m], m + k)  
        for m in range(N):
            if 2 * m + 1 < N - 1:
                circuit.cnot(2 * m + 1 + k  , 2 * m + 2 + k)
    return circuit

def vqc_qst(depth, circuit, thetas, N, k): #VQC definition.
    for y in range(depth):
        circuit = block_qst(y, depth, circuit, thetas, N, k)
    return circuit
    
def cost(thetas, depth, N): #Computation of the cost function
    seedT = 42
    seed_H = 10
    cost_circuit = QuantumCircuit(1 + 2 * N, 1)
    U_Haar = random_unitary(2 ** N, seed = seed_H) #Define Haar target
    lista = []
    for i in range(N):
        lista.append(1 + i)
    cost_circuit.unitary(U_Haar, lista)
    cost_circuit = vqc_qst(depth, cost_circuit, thetas,  N, N + 1)
    aer_sim = Aer.get_backend('aer_simulator')
    swap_test_circuit = cost_circuit.copy()
    swap_test_circuit.h(0)
    for i in range(N):
            swap_test_circuit.cswap(0, i + 1, 1 + N + i)
    swap_test_circuit.h(0)
    swap_test_circuit.measure(0, 0)
    T_swap_test_circuit = transpile(swap_test_circuit, aer_sim,
    seed_transpiler = seedT)
    shots = 1000
    results = aer_sim.run(T_swap_test_circuit, shots = shots).result()
    try:
        frequency = results.get_counts()['1'] / shots 
    except:
        frequency = 0
    else:
        frequency = results.get_counts()['1'] / shots 
    if frequency > 0.49:
        frequency = 0.49
    overlap = np.sqrt(1 - 2 * frequency)
    return 1 - overlap

def gradient(thetas, N, depth): #Gradient computation
	diff = np.pi / 2
	grad = np.zeros((depth, N))
	for i in range(depth):
            for m in range(N):
                    params = np.copy(thetas)
                    losses= []
                    for p in range(2):
                          params[i][m] += diff * (-1) ** p - diff * p
                          losses.append(cost(params, depth, N))
                    grad[i][m] = (losses[0] - losses[1]) / 2                
	return  grad
###################################################

#Adam implementation
###################################################
def QST(N, depth, iterations):
    alpha = 0.1
    b_1 = 0.8            
    b_2 = 0.999
    epsilon = 10 ** (-8)
    s = np.zeros((depth, N))
    v = np.zeros((depth, N))
    s_hat = np.zeros((depth, N))
    v_hat = np.zeros((depth, N))
    thetas = np.zeros((depth, N)) #Parameters to be trained
    for i in range(depth):
        for m in range(N):
            thetas[i][m] = np.random.uniform(0, 2 * np.pi)
    for w in range(iterations):
        grad = gradient(thetas, N, depth)
        for i in range(depth):
            for m in range(N):
                s[i][m] = b_1 * s[i][m] + (1 - b_1) *  grad[i][m]
                v[i][m] = b_2 * v[i][m] + (1 - b_2) * grad[i][m] ** 2
                s_hat[i][m] = s[i][m] / (1 - b_1 ** (w + 1))
                v_hat[i][m] = v[i][m] / (1 - b_2 ** (w + 1))
                thetas[i][m] -= alpha * s_hat[i][m] / (np.sqrt(v_hat[i][m]) + epsilon)   
    final_cost = cost(thetas, depth, N)
    return  final_cost, thetas
###################################################

#Calling QST(N, depth, iterations)
###################################################
iterations = 10
depth = 2
N = 1
final_cost, thetas= QST(N, depth, iterations)
np.save('thetas.txt', thetas)
###################################################