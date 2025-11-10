#Importing Libraries
##############################################################
import numpy as np
from qiskit import QuantumCircuit, transpile
import qiskit.quantum_info as qi
from qiskit import ClassicalRegister, QuantumRegister

from qiskit.circuit.add_control import add_control
from qiskit.result import marginal_counts
from qiskit_aer import Aer
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.quantum_info import random_unitary
import multiprocessing
from multiprocessing import Pool
##############################################################

#Four-terms shift rule constants: parameters definition.
##############################################################
four_term_psr = {'alpha': np.pi / 2 - np.pi / 4,
    'beta' : np.pi / 2 + np.pi / 4,
    'd_plus' : (np.sqrt(2) + 1) / (2 * np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (2 * np.sqrt(2))}
##############################################################

#Auxilliary functions 
##############################################################
def dec_to_bin(x, N): #From decimal to binary N-bits string.
    a = np.copy(x)
    result = []
    for i in range(N):
        if a >= 2 ** (N - 1 - i):
            result.append(1)
            a -= 2 ** (N - 1 - i)
        else:
            result.append(0)
    return result

def block(s, N, circuit, thetas): #Single block of the VQC.
    aux = 0
    for p in range(N):
                circuit.rz(thetas[3 * s * N + p * 3], p)
                circuit.ry(thetas[3 * s * N + p * 3 + 1], p)
                circuit.rz(thetas[3 * s * N + p * 3 + 2], p)
    if s % 2 != 0:
        aux = 1
    for p in range(N):
        if 2 * p < N - (1 + aux):
            circuit.cnot(2 * p + aux, 2 * p + 1 + aux)
    return circuit

def init(circuit, N, pos): #Iitialization of the circit to specific basis state.
    string = dec_to_bin(pos, N)
    for u in range(N):
        if string[u] == 1:
            circuit.x(u)
    return circuit

def cohe_sum_1(circuit, X, N): #First step in creating the desired coherent supersposition.
        CW = add_control(X, ctrl_state = '1', label="CW", num_ctrl_qubits = 1)
        circuit.h(N)
        lista = [N]
        for j in range(N):
            lista.append(j)
        circuit.append(CW, lista)
        return circuit

def cohe_sum_2(circuit, thetas, N, depth): #Second step in creating the desired coherent superposition.
    aux_circ = QuantumCircuit(N)
    for m in range(depth):
        aux_circ = block(m, N, aux_circ, thetas)
    for u in range(N):
        aux_circ.rz(thetas[3 * depth * N + u * 3], u)
        aux_circ.ry(thetas[3 * depth * N + u * 3 + 1], u)
        aux_circ.rz(thetas[3 * depth * N + u * 3 + 2], u)
    J = qi.Operator(aux_circ)
    U_Gate=UnitaryGate(J  ,label="Learn_U")            
    CH = add_control(U_Gate, num_ctrl_qubits = 1, ctrl_state = 0, label = "U_L")
    lista = [N]
    for j in range(N):
        lista.append(j)
    circuit.append(CH, lista)
    circuit.h(N)
    return circuit

def circuit_prep(thetas, N, X, depth, pos): #Prepare the whole circuit before for cost function evaluation.
    cost_circuit = QuantumCircuit(N + 1, 1)
    cost_circuit = init(cost_circuit, N, pos)
    cost_circuit = cohe_sum_1(cost_circuit, X, N)
    cost_circuit = cohe_sum_2(cost_circuit, thetas, N, depth)
    return cost_circuit
                       
def sum_frequencies_all(vec_circuit, N): #Evaluate the observed frequency at each different initialization.
    aer_sim = Aer.get_backend('aer_simulator')
    seed_T = 42
    shots = 1000
    lista = []
    for y in range(2 ** N):
        qpe2 = vec_circuit[y]
        qpe2.measure(N, [0])
        T_aux_circuit = transpile(qpe2, aer_sim, seed_transpiler = seed_T)
        results = aer_sim.run(T_aux_circuit, shots = shots).result()
        try:
            frequency = marginal_counts(results, indices = [0]).get_counts()['0'] / shots
        except:
            frequency = 1
        real = 2 * frequency - 1
        lista.append(1 - real)
    return sum(lista) / (2 ** N)
##############################################################

#Gradient computation
##############################################################
def grad_loss(thetas, N, X, depth, i):
    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas3, thetas4 = thetas.copy(), thetas.copy()
    thetas1[i] += four_term_psr['alpha']
    thetas2[i] -= four_term_psr['alpha']
    thetas3[i] += four_term_psr['beta']
    thetas4[i] -= four_term_psr['beta']
    qc1=[]
    qc2=[]
    qc3=[]
    qc4=[]
    for y in range(2 ** N):
        qc1.append(circuit_prep(thetas1, N, X, depth, y))
        qc2.append(circuit_prep(thetas2, N, X, depth, y))
        qc3.append(circuit_prep(thetas3, N, X, depth, y))
        qc4.append(circuit_prep(thetas4, N, X, depth, y))
    A = sum_frequencies_all(qc1, N)
    B = sum_frequencies_all(qc2, N)
    C = sum_frequencies_all(qc3, N)
    D = sum_frequencies_all(qc4, N)
    
    return (four_term_psr['d_plus'] * (A - B) - four_term_psr['d_minus'] * (C - D))

def task_wrapper(args): #Function needed for the parallellization of the gradient computation.
    return grad_loss(*args)
##############################################################

#Adam implementation
##############################################################
def Process(X, N, iterations, depth, cores):
    alpha = 0.1
    b_1 = 0.8
    b_2 = 0.999
    epsilon = 10 ** (-8)
    w = np.zeros(depth * N * 3 + 3 * N)
    v = np.zeros(depth * N * 3 + 3 * N)
    w_hat = np.zeros(depth * N * 3 + 3 * N)
    v_hat = np.zeros(depth * N * 3 + 3 * N)
    thetas = np.ones(depth * N * 3 + 3 * N)
    for i in range(depth * N * 3 + 3 * N):
        thetas[i]= np.random.uniform(0, 2 * np.pi)
    qk=[]
    for t in range(iterations):
        print("Iteration: "+str(t)+"...")
        i_vec=[]
        for j in range(len(thetas)):
            i_vec.append(j)
        args = [(thetas, N, X, depth, i_vec[i]) for i in range(len(i_vec))]
        with Pool(cores) as pool:
            g_dat = pool.map(task_wrapper, args)
            pool.close()
            pool.join()
        grad = g_dat
        for i in range(len(thetas)):
            w[i] = b_1 * w[i] + (1 - b_1) * grad[i]
            v[i] = b_2 * v[i] + (1 - b_2) * grad[i] ** 2
            w_hat[i] = w[i] / (1 - b_1 ** (t + 1))
            v_hat[i] = v[i] / (1 - b_2 ** (t + 1))
            thetas[i] -= alpha * w_hat[i] / (np.sqrt(v_hat[i]) + epsilon)
        for y in range(2 ** N):
            qk.append(circuit_prep(thetas, N, X, depth, y))
    circ = QuantumCircuit(N)
    for m in range(depth):
            circ = block(m, N, circ, thetas)  
    for i in range(N):
            circ.rz(thetas[3 * depth * N + i * 3], i)
            circ.ry(thetas[3 * depth * N + i * 3 + 1], i)
            circ.rz(thetas[3 * depth * N + i * 3 + 2], i) 
    M = qi.Operator(circ)
    Mu=UnitaryGate(M, label = "Learn_U")   
    return M.to_matrix(), qk
##############################################################

#Calling Process(X, N, iterations, depth, cores)
##############################################################
if __name__ == "__main__":
    cores= 1 #If set > 1, it allows for the parallellization of the gradient computation.
    N = 1
    iterations = 2
    depth = 3
    seed_H = 42
    print("Starting...")
    U_Haar = random_unitary(2 ** N, seed = seed_H) #Define Haar target
    V = UnitaryGate(U_Haar)
    Learn_U, qk = Process(V, N, iterations, depth, cores)
    costs=[]
    for j in range(iterations - 1):
        costs.append(sum_frequencies_all(qk[j * (2 ** N):(j + 1) * (2 ** N)], N)) 

    print("Cost function from iter to iter+1 from Process Tomography:")
    print(costs)
    np.save('Learnt_Unitary.txt', Learn_U)
    np.save('Costs.txt', costs)
##############################################################
