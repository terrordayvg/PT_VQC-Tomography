#Importing libraries
##############################################################
import numpy as np
from qiskit.quantum_info import random_unitary
from qiskit import ClassicalRegister, QuantumRegister, execute
from qiskit import assemble, Aer, IBMQ
from qiskit.circuit.add_control import add_control
from qiskit.result import marginal_counts
import qiskit.quantum_info as qi
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import QFT

#Auxilliary functions
##############################################################
def Inv_QFT(circuit, A): #Performs inverse QFT to first "A" qubits of the system.
    listing = []
    for e in range(A):
        listing.append(e)
    qft_i = QFT(num_qubits = A, approximation_degree = 0, do_swaps = False, inverse = True, insert_barriers = False, name = 'qfti').to_gate()
    circuit.append(qft_i, listing)
    return circuit

def pos_eigen(y, eigens, A): #Decides whcih learnt eingenvector to send
    pos = 0
    counts = 0
    diff = 10 ** 7
    for x in eigens:
        if abs( y - (2 ** A) * x) <= 2 ** A / 2 :
            if diff > abs( y - (2 ** A) * x ):
                diff = abs( y - (2 ** A) * x)
                pos = counts
        else:
            if diff > 2 ** A - abs(y - ( 2 ** A) * x):
                diff = 2 ** A - abs(y - (2 ** A) * x)
                pos = counts
        counts += 1
    return pos

def dec_to_bin(x, A):
    result = np.zeros(A, dtype = int)
    for g in range(A):
        if x >= 2 ** (A-1-g):
            result[-1-g] = 1
            x -= 2 ** (A-1-g)
    return result

def svd_init(circuit, y, A, T, M, eigens): #Prepares the selected eigenvector.
    pos = pos_eigen(y, eigens, A)
    approx = eigens[pos]
    string = dec_to_bin(pos, T)
    for i in range(T):
        if string[i] == 1:
            circuit.x(A + i)
    listing = []
    for i in range(T):
        listing.append(A + i)
    circuit.append(M, listing)
    return circuit, approx

def cyclic(outcome1, outcome2, A): #Outputs cyclic between generation and verification stages.
    result = abs(outcome1 - outcome2)
    if result  > 2 ** A / 2:
        result = (2 ** A - abs(outcome1 - outcome2))
    return result
##############################################################

#Outputs cyclic deviation for (depending on 'w') a SVD attack, a random attack or the user.
##############################################################
def Forgery_SVD(U, T, A, w):
                                M = np.loadtxt('file_path', dtype= complex) #Load unitary matrix encoding the different singular vectors
                                M = UnitaryGate(M)                          #Load the different singular values
                                eigens = np.loadtxt('file_path', dtype= complex)
                                seedT = 10
                                simulator = Aer.get_backend("qasm_simulator")
                                circuit = QuantumCircuit(A + T, 2 * A)
                                CU2 = add_control(U, 1, ctrl_state = 1, label="CHar")
                                for qubit in range(A):
                                    circuit.h(qubit)
                                repetitions=1
                                for counting_qubit in range(A):
                                    listing = [A-1-counting_qubit]
                                    for i in range(T):
                                        listing.append(A + i)
                                    for j in range(repetitions):
                                            circuit.append(CU2, listing)
                                    repetitions *= 2
                                circuit = Inv_QFT(circuit,A) 
                                list_q = []
                                list_c_1 = []
                                list_c_2 = []
                                for i in range(A):
                                    list_q.append(i)
                                    list_c_1.append(i)
                                    list_c_2.append(A + i)
                                circuit.measure(list_q,list_c_1)
                                if w != 2:                                  #This 'if' condition apply to SDV and random attacks
                                    t_circuit = transpile(circuit, simulator, seed_transpiler = seedT)
                                    result1 = simulator.run(t_circuit, shots = 1).result()
                                    simp_counts1 = marginal_counts(result1, indices = list_c_1).get_counts()
                                    key1, value1 = list(simp_counts1.items())[0]
                                    outcome1 = int(key1, 2)
                                    attack_circuit = QuantumCircuit(A + T, A)
                                    if w == 0:                              #SVD attack preparation
                                        attack_circuit, approx = svd_init(attack_circuit, outcome1, A, T, M, eigens) 
                                    for qubit in range(A):
                                        attack_circuit.h(qubit)
                                    repetitions=1
                                    for counting_qubit in range(A):
                                        listing = [A - 1 - counting_qubit]
                                        for i in range(T):
                                            listing.append(A + i)
                                        for j in range(repetitions):
                                                attack_circuit.append(CU2, listing)
                                        repetitions *= 2
                                    attack_circuit = Inv_QFT(attack_circuit, A)
                                    attack_circuit.measure(list_q,list_c_1)
                                    t_attack_circuit = transpile(attack_circuit, simulator)
                                    result2 = simulator.run(t_attack_circuit, shots = 1).result()
                                    simp_counts2 = marginal_counts(result2, indices = list_c_1).get_counts()
                                    key2, value2 = list(simp_counts2.items())[0]
                                    outcome2 = int(key2, 2)
                                else:                                   #This "else" conditions preceeds the user-server interaction at verification.
                                    circuit.reset(list_auxq)
                                    for qubit in range(A):
                                        circuit.h(q[qubit])
                                    repetitions = 1
                                    for counting_qubit in range(A):
                                        listing = [q[A - 1 - counting_qubit]]
                                        for i in range(T):
                                                listing.append(A +  i)
                                        for j in range(repetitions):
                                                circuit.append(CU2, listing)
                                        repetitions *= 2
                                    circuit = Inv_QFT(circuit, A) 
                                    circuit.measure(list_q, list_c_2)
                                    t_circuit = transpile(circuit, simulator ,seed_transpiler = seedT)
                                    result = simulator.run(t_circuit, shots = 1).result()
                                    simp_counts1 = marginal_counts(result, indices = list_c_1).get_counts()
                                    key1, value1 = list(simp_counts1.items())[0]
                                    outcome1=int(key1, 2)
                                    simp_counts2 = marginal_counts(result, indices = list_c_2).get_counts()
                                    key2, value2 = list(simp_counts2.items())[0]
                                    outcome2=int(key2, 2)
                                return cyclic(outcome1, outcome2, A)        #Returns cyclic distance between generated and verification classical outcomes for the chosen type of interaction ('w').
##############################################################

#Calling Forgery_SVD(U, T, A, eigens, M, w)
##############################################################
A = 2
T = 1
w = 0 #0: qst_attack, 1: random_attack, 2: user
U_Haar = random_unitary(2 ** T) #Haar-random unitary defining the QPUF
W_Haar = UnitaryGate(U_Haar)
deviation = Forgery_SVD(W_Haar, T, A, w)
##############################################################
