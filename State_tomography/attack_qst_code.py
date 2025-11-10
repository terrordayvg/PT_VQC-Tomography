#Importing libraries
##############################################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.quantum_info as qi
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit.add_control import add_control
from qiskit.result import marginal_counts
from qiskit import assemble
from qiskit_aer import Aer

from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
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

def cyclic(outcome1, outcome2, A):  #Returns cyclic distance between outcomes
    result = abs(outcome1 - outcome2) 
    if result  > 2 ** A / 2:
        result = (2 ** A - abs(outcome1 - outcome2))
    return result

def dec_to_bin(x, A):
    result = np.zeros(A, dtype = int)
    for g in range(A):
        if x >= 2 ** (A-1-g):
            result[-1-g] = 1
            x -= 2 ** (A-1-g)
    return result

def bin_to_dec(x, A):
    result = 0
    for g in range(A):
        result += x[g] * 2 ** (A - 1 - g)
    return result

def block_qst(d,  depth, circuit, thetas, T, r, outcome): #Single block of the VQC.
    if d == depth - 1:
            for q in range(T):
                    circuit.rx(thetas[outcome][d][q], q + r)
    elif d % 2 == 0:
                for q in range(T):
                    circuit.ry(thetas[outcome][d][q], q + r)   
                for q in range(T):
                    if 2*q < T -1:
                            circuit.cnot(2 * q + r  , 2 * q + 1 + r)
    else:
                for q in range(T):
                        circuit.rx(thetas[outcome][d][q], q + r)  
                for q in range(T):
                        if 2*q + 1 < T -1:
                            circuit.cnot(2 * q + 1 + r , 2 * q + 2 + r)
    return circuit

def vqc_qst(depth, circuit, thetas, T, r, outcome): #Creates the shole VQC.
    for d in range(depth):
            circuit = block_qst(d, depth, circuit, thetas, T, r, outcome)
    return circuit

def gate_vqc_qst(depth, thetas, T, r, outcome, A, opt): #Creates a Quantum Gate out of the VQC.
    circuit = QuantumCircuit(2 * T + A + opt)
    circuit = vqc_qst(depth, circuit, thetas, T, r, outcome)
    Op =  qi.Operator(circuit)
    H = UnitaryGate(Op)
    return H
    
def cost(freqs, A): #Computation of the cost function
    a = np.zeros(2 ** A)
    for f in range(2 **  A): 
        if freqs[f] > 0.49:
            freqs[f] = 0.49
        a[f] =  1 - np.sqrt(1 - 2 * freqs[f])
    return a

def QPUF(X, circuit, T, A, reg): #Performs the QPUF operation and records the outcome in the ancilla register via "A" c-bits and an extra qubit. For learning purposes.
                                aer_sim = Aer.get_backend('aer_simulator')
                                seedT = 10
                                CU2 = add_control(X, 1, ctrl_state = 1, label = 'Control')
                                for qubit in range(A):
                                    circuit.h(qubit)
                                repetitions=1
                                for counting_qubit in range(A):
                                    listing = [A - 1 - counting_qubit]
                                    for h in range(T):
                                        listing.append(A + h)
                                    for h in range(repetitions):
                                            circuit.append(CU2, listing)
                                    repetitions *= 2
                                circuit = Inv_QFT(circuit, A) 
                                list_aux1 = []
                                for h in range(A):
                                    list_aux1.append(h)
                                circuit.measure(list_aux1, list_aux1)
                                for v in range(2 ** A):
                                        aux = dec_to_bin(v, A)
                                        aux_v = []
                                        for h in range(A):
                                            aux_v.append(aux[- 1 - h])
                                        entry = bin_to_dec(aux_v, A)
                                        circuit.x(- 2).c_if(reg, v)

                                        circuit.measure(- 2, A + entry)
                                        circuit.x(- 2).c_if(reg, v)  
  
                                return circuit

def QPUF_veri(X, circuit, T, A, reg, flag, target): #Performs the QPUF operation and records the outcome in the ancilla register via "A" c-bits and an extra qubit. For generation and verification purposes.
                                aer_sim = Aer.get_backend('aer_simulator')
                                seedT = 10
                                CU2 = add_control(X, 1, ctrl_state = 1, label = 'Control')
                                for qubit in range(A):
                                    circuit.h(qubit)
                                repetitions = 1
                                for counting_qubit in range(A):
                                    listing = [A - 1 - counting_qubit]
                                    for h in range(T):
                                        listing.append(A + target + h)
                                    for h in range(repetitions):
                                            circuit.append(CU2, listing)
                                    repetitions *= 2
                                circuit = Inv_QFT(circuit, A) 
                                list_aux_1 = []
                                list_aux_2 = []
                                for h in range(A):
                                    list_aux_1.append(h)
                                    list_aux_2.append(A + 2 ** A + h) 
                                if flag == 0:    
                                    circuit.measure(list_aux_1,list_aux_1)
                                else:
                                    circuit.measure(list_aux_1,list_aux_2)
                                if flag == 0:    
                                    for v in range(2 ** A):
                                                aux = dec_to_bin(v, A)
                                                aux_v = []
                                                for h in range(A):
                                                    aux_v.append(aux[- 1 - h])
                                                entry = bin_to_dec(aux_v, A)
                                                circuit.x(-1).c_if(reg, v)
                                                circuit.measure(- 1, A + entry)
                                                circuit.x(-1).c_if(reg, v)
                                flag += 1
                                return circuit, flag

def swap(circuit, A, T, frequs, totalis): #Performs swap test between post-measurement state and corresponding VQC.
                                seedT = 12
                                aer_sim = Aer.get_backend('aer_simulator')
                                circuit.h(-1)
                                for u in range(T):
                                    circuit.cswap(- 1, A + u, A + T + u)
                                circuit.h(-1)
                                circuit.measure(-1,  A  + 2 ** A)
                                T_swap_test_circ = transpile(circuit, aer_sim, seed_transpiler = seedT)
                                result_s = aer_sim.run(T_swap_test_circ, shots = 1).result()
                                count = marginal_counts(result_s, indices=[A + 2 ** A]).get_counts() 
                                key1, value_aux = list(count.items())[0]
                                outcome_aux = int(key1, 2)
                                if outcome_aux == 1:
                                    frequs += 1
                                totalis += 1
                                return circuit, frequs, totalis

def check(b, PLAUSIBLE): #Checks whether an outcome has appeared in at least one QPUF query.
    a = 0
    for x in PLAUSIBLE:
        if b == x:
            a = 1
    return a 

def gradient(depth, A, thetas, T, X): #Computation of the gradient.
    diff = np.pi / 2
    dq = 2 * T + A + 2
    dc = A  + 1 + 2 ** A
    grad = np.zeros((2 ** A, depth, T))
    for c in range(depth):
            for l in range(T):
                    for z in range(2 ** A):
                            losses = []
                            params = np.copy(thetas)
                            for p in range(2):
                                params[z][c][l] += diff * (-1) ** p - diff * p
                                frecu = 0
                                totalu = 0
                                losses.append([])
                                acounts = 0
                                while acounts < 25 * 2 ** A:
                                                q_regg = QuantumRegister(dq, 'quantum a')
                                                c_regg = ClassicalRegister(dc, 'classical a')
                                                qpe2 = QuantumCircuit(q_regg,c_regg)
                                                qpe2 = QPUF(X, qpe2, T, A,c_regg)
                                                listy = []
                                                for b in range(2 * T + A + 2):                                                    

                                                    listy.append(b)
                                                for o in range(2 ** A):
                                                    qpe2.append(gate_vqc_qst(depth, params, T, T + A, o, A, 2), listy).c_if(c_regg[A + o], 1)
                                                qpe2, frecu, totalu = swap(qpe2, A,T, frecu, totalu)
                                                acounts += 1

                                frecu = frecu / totalu
                                if frecu > 0.49:
                                    frecu = 0.49
                                losses[p] = frecu
                                
                            grad[z][c][l] = (losses[0] - losses[1]) / 2        
    return  grad

def Adam_QPUF(T, A, X, depth, iterations): #Learns the QPUF post-measurement states via Adam optimizer. 
        cost = 1
        alpha = 0.1
        b_1 = 0.8           
        b_2 = 0.999
        epsilon = 10 ** (-8)
        aer_sim = Aer.get_backend('aer_simulator')
        dq = 2 * T + A + 2 
        dc = A  + 1 + 2 ** A
        m = np.zeros((2 ** A, depth, T))
        v = np.zeros((2 ** A, depth, T))
        m_hat = np.zeros((2 ** A, depth, T))
        v_hat = np.zeros((2 ** A, depth, T))
        thetas = np.zeros((2 ** A, depth, T))
        for y in range(2 ** A):
            for i in range(depth):
                for j in range(T):
                    thetas[y][i][j] = np.random.uniform(0, 2 * np.pi)
        for k in range(iterations):
            print("Iteration "+str(k)+"...")
            freq = 0
            total = 0   
            counts = 0
            while counts < 25 * 2 ** A:
                                q_reg = QuantumRegister(dq, 'quantum a')
                                c_reg = ClassicalRegister(dc, 'classical a')
                                qpe2 = QuantumCircuit(q_reg, c_reg)
                                qpe2 = QPUF(X, qpe2, T, A, c_reg)
                                listing = []
                                for b in range(2 * T + A + 2):
                                    listing.append(b) 
                                for o in range(2 ** A):
                                    qpe2.append(gate_vqc_qst(depth, thetas, T, T + A, o, A, 2), listing).c_if(c_reg[A + o], 1)

                                qpe2, freq, total = swap(qpe2, A, T, freq, total)
                                counts += 1
            freq = freq / total
            if freq > 0.49:
                    freq = 0.49
            cost = freq
            grad = gradient(depth,  A, thetas, T, X)
            for y in range(2**A):
                    for i in range(depth): 
                            for j in range(T):
                                m[y][i][j] = b_1 * m[y][i][j] + (1 - b_1) * grad[y][i][j]
                                v[y][i][j] = b_2 * v[y][i][j] + (1 - b_2) * grad[y][i][j] ** 2
                                m_hat[y][i][j] = m[y][i][j] / (1 - b_1 ** (k + 1))
                                v_hat[y][i][j] = v[y][i][j] / (1 - b_2 ** (k + 1))
                                thetas[y][i][j] -= alpha * m_hat[y][i][j] / (np.sqrt(v_hat[y][i][j]) + epsilon) 
        return thetas, cost
##############################################################

#Outputs the deviation between generation and verification outcomes fo for either (w) a QST attacker, a random attacker or a user.
##############################################################
def Forgery_QST(X, T, A, w, thetas): 
                                simulator = Aer.get_backend("qasm_simulator")
                                dq = A + 2 * T + 1
                                dc =  2 * A  +  2 ** A
                                q_reg = QuantumRegister(dq, 'quantum a')
                                c_reg = ClassicalRegister(dc, 'classical a')
                                circuit= QuantumCircuit(q_reg,c_reg)
                                list_1 = []
                                list_2 = []
                                for h in range(A):
                                    list_1.append(h)
                                    list_2.append(A + 2 ** A + h)
                                flag = 0
                                circuit, flag = QPUF_veri(X, circuit, T, A, c_reg, flag, 0)
                                if w != 2:
                                    circuit.reset(list_1)
                                    if w == 0:
                                        listy = []
                                        for b in range(2 * T + A + 1):
                                            listy.append(b)
                                        for o in range(2 ** A):

                                            circuit.append(gate_vqc_qst(depth, thetas, T, T + A, o, A, 1), listy).c_if(c_reg[A + o], 1)

                                        circuit, flag = QPUF_veri(X, circuit, T, A, c_reg, flag, T)
                                    elif w == 1:
                                        circuit, flag = QPUF_veri(X, circuit, T, A, c_reg, flag, T)
                                else:
                                    circuit.reset(list_1)
                                     
                                    circuit, flag = QPUF_veri(X, circuit, T, A, c_reg, flag, 0)
                                t_circuit = transpile(circuit, simulator ,seed_transpiler = 10)
                                #simulator = AerSimulator(method="automatic")  

                                result = simulator.run(t_circuit, shots = 1).result()
                                simp_counts1 = marginal_counts(result, indices = list_1).get_counts()
                                key1, value_1 = list(simp_counts1.items())[0]
                                outcome_1 = int(key1, 2)
                                simp_counts2 = marginal_counts(result, indices = list_2).get_counts()                                   
                                key2, value_2 = list(simp_counts2.items())[0]
                                outcome_2 = int(key2, 2)
                                return cyclic(outcome_1, outcome_2, A)
##############################################################
if __name__ == "__main__":
#Calling Forgery_QST(W_Haar, T, A, w, thetas)
##############################################################
    A = 2
    T = 1
    w = 0 #0: qst_attack, 1: random_attack, 2: user
    U_Haar = random_unitary(2 ** T) #Haar-random unitary defining the QPUF
    W_Haar = UnitaryGate(U_Haar)
    depth = 2
    iterations = 1
    print("Processing...")
    thetas, cost = Adam_QPUF(T, A, W_Haar, depth, iterations)
    print("Starting forgery...")
    deviation = Forgery_QST(W_Haar, T, A, w, thetas)
##############################################################