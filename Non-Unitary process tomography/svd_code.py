# Importing libraries
##############################################################
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import random_unitary
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit.add_control import add_control
from qiskit.result import marginal_counts
import qiskit.quantum_info as qi
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit import assemble
from qiskit_aer import Aer
import multiprocessing
from multiprocessing import Pool
##############################################################

#Auxilliary functions
##############################################################
def dec_to_bin(x, N): #From decimal to binary N-bits string
    result = []
    for g in range(N):
        if x >= 2 ** (N - 1 - g):
            result.append(1)
            x -= 2 ** (N - 1 - g)
        else:
            result.append(0)
    return result
    
def block_svd(circuit, y, N, alphas, k): #Single block of the VQC.
        for h in range(N): 
                circuit.rz(alphas[3 * N * y + 3 * h], h + 1 + k * N)
                circuit.ry(alphas[3 * N * y + 3 * h + 1], h + 1 + k * N)
                circuit.rz(alphas[3 * N * y + 3 * h + 2] , h + 1 + k * N)
        if y % 2 == 0:
            for h in range(N - 1):
                    circuit.cnot(h + 1 + k * N, h + 2 + k * N)
        else:
            for h in range(N - 2):
                    circuit.cnot(h + 2 + k * N, h + 3 + k * N)
        return circuit


def block_svd_def(circuit, y, N, alphas): #Single block of the VQC.
        for h in range(N): 
                circuit.rz(alphas[3 * N * y + 3 * h], h)
                circuit.ry(alphas[3 * N * y + 3 * h + 1], h)
                circuit.rz(alphas[3 * N * y + 3 * h + 2] , h)
        if y % 2 == 0:
            for h in range(N - 1):
                    circuit.cnot(h + k * N, h + 1)
        else:
            for h in range(N - 2):
                    circuit.cnot(h + 1 + k * N, h + 2)
        return circuit

def cont_block_svd(circuit, y, N, alphas): #Controlled version of the single block of the VQC.
    for h in range(N):
            circuit.crz(alphas[3 * N * y + 3 * h], 0, h + 1)
            circuit.cry(alphas[3 * N * y + 3 * h + 1], 0, h + 1)
            circuit.crz(alphas[3 * N * y + 3 * h + 2] , 0, h + 1)
    if y % 2 == 0:
        for h in range(N - 1):
                circuit.toffoli(0, h + 1, h + 2)
    else:
        for h in range(N - 2):
                circuit.toffoli(0, h + 2, h + 3)
    return circuit

def complex_to_phase(eigens): #From a norm-1 complex number to its phase angle in [0, 2pi).
    lista = []
    for x in eigens:
        real = x.real
        imag = x.imag
        if real > 10 ** (- 4):
            phase = np.arctan(imag/real)
            if real < 0:
                phase += np.pi
            if phase < 0:
                phase += 2 * np.pi
        else:
            if imag > 0:
                phase = np.pi / 2
            else:
                phase = 3 / 2 * np.pi
        lista.append(phase)
    return lista
    
def swap_test(circuit, N):  
        aer_sim = Aer.get_backend('aer_simulator')
        circuit.h(0)
        for u in range(N):
            circuit.cswap(0, N + u + 1, u + 1)
        circuit.h(0)
        circuit.measure(0, 0)
        T_swap_test_circuit = transpile(circuit, aer_sim)
        shots = 1000
        result = aer_sim.run(T_swap_test_circuit, shots = shots).result()
        try:
            counts =result.get_counts()['1'] / shots
        except:
            counts = 0
        if counts > 0.49:
            counts = 0.49
        overlap = np.sqrt(1 - 2 * counts)
        return overlap
    
def loss_svd(alphas, N, depth, U): #Cost computation
    aer_sim = Aer.get_backend('aer_simulator')
    seedT = 50
    d = 2 ** N
    cost = 0
    llista = []
    for i in range(d):
            h_test = QuantumCircuit(2 * N + 1, 1)
            string = dec_to_bin(i, N)
            for w in range(N):
                if string[w] == 1:
                    h_test.x(w + 1)
                    h_test.x(w + 1 + N)
            for y in range(depth):
                h_test = block_svd(h_test, y, N, alphas, 0)
            lista1 = []
            for w in range(N):
                lista1.append(w + 1)
            h_test.append(U, lista1)
            for y in range(depth):
                h_test = block_svd(h_test, y, N, alphas, 1)
            overlap = swap_test(h_test, N)
            cost += (1 - overlap) / d
    return cost
    
def singu(N, U, alphas, check_real, check_imag, depth): #Returns the learnt singular value so far.
    d = 2 ** N
    singular = np.zeros(d, dtype = complex)
    Cgate = add_control(U, ctrl_state = '1', label = "CHar", num_ctrl_qubits = 1)
    aer_sim = Aer.get_backend('aer_simulator')
    pos_real = np.zeros(d)
    pos_imag = np.zeros(d)
    L_real= np.zeros(d)
    L_imag = np.zeros(d)
    for i in range(d):
            h_test_real = QuantumCircuit(N + 1, 1)
            h_test_real.h(0)
            h_test_imag = QuantumCircuit(N + 1, 1)
            h_test_imag.h(0)
            h_test_imag.p(3 * np.pi / 2, 0)
            string = dec_to_bin(i, N)
            for w in range(N):
                    if string[w] == 1:
                        h_test_real.x(w + 1)
                        h_test_imag.x(w + 1)
            for y in range(depth):
                    h_test_real =  cont_block_svd(h_test_real, y, N, alphas)
                    h_test_imag =  cont_block_svd(h_test_imag, y, N, alphas)
            lista_aux_real = []
            lista_aux_imag = []
            for y in range(N + 1):
                lista_aux_real.append(y)
                lista_aux_imag.append(y)
            h_test_real.append(Cgate, lista_aux_real)
            h_test_imag.append(Cgate, lista_aux_imag)
            h_test_real.x(0)
            h_test_imag.x(0)
            for y in range(depth):
                    h_test_real =  cont_block_svd(h_test_real, y, N, alphas)
                    h_test_imag =  cont_block_svd(h_test_imag, y, N, alphas)
            h_test_real.x(0)
            h_test_real.h(0)
            h_test_imag.x(0)
            h_test_imag.h(0)
            h_test_real.measure(0, 0)
            h_test_imag.measure(0, 0)
            T_h_test_real = transpile(h_test_real, aer_sim)
            T_h_test_imag = transpile(h_test_imag, aer_sim)
            shots = 1000
            results_real = aer_sim.run(T_h_test_real, shots=shots).result()
            results_imag = aer_sim.run(T_h_test_imag, shots=shots).result()
            try:
                    pos_real[i] = results_real.get_counts()['0'] / shots
                    L_real[i] += 2 * pos_real[i] - 1
            except:
                    if check_real[i] > 0.5:
                        pos_real[i] = 1
                    else:
                        pos_real[i] = 0
                    
                    L_real[i] += 2 * pos_real[i] - 1
            
            try:
                    pos_imag[i] = results_imag.get_counts()['0'] / shots
                    L_imag[i] += 2 * pos_imag[i] - 1
            except:
                    if check_imag[i] > 0.5:
                        pos_imag[i] = 1
                    else:
                        pos_imag[i] = 0
                    L_imag[i] += 2 * pos_imag[i] - 1
            
            singular[i] = complex(L_real[i], L_imag[i])
    return singular, pos_real, pos_imag

def rot_derivative(U, N, depth, alphas, i):
    diff = np.pi/2
    angles = np.copy(alphas)
    losses = []
    for p in range(2):
        angles[i] += diff * (-1) ** p - diff * p
        cost =  loss_svd(angles, N, depth, U)
        losses.append(cost)
    deriv = (losses[0] - losses[1]) / 2
    return deriv

def task_wrapper(args): 
    return rot_derivative(*args)
##############################################################

#Adam implementation
##############################################################
def SVD(N, depth, U, iterations, cores):
    d = 2 ** N
    alpha = 0.1
    b_1 = 0.8
    b_2 = 0.999
    epsilon = 10 ** (-8)
    alphas = np.zeros(3 * N * depth)
    m1 = np.zeros(3 * N * depth)
    v1 = np.zeros(3 * N * depth)
    m_hat1 = np.zeros(3 * N * depth)
    v_hat1 = np.zeros(3 * N * depth)
    m2 = np.zeros(3 * N * depth)
    v2 = np.zeros(3 * N * depth)
    m_hat2 = np.zeros(3 * N * depth)
    v_hat2 = np.zeros(3 * N * depth)
    for i in range(3 * N * depth):
        alphas[i] = np.random.uniform(0, 2 * np.pi)
    check_real = np.ones(d) #In order to deal with the error happening one all counts fall into same outcome '0' or '1'.
    check_imag = np.ones(d)
    cost = 1
    flag = 0
    for ites in range(iterations):
        print("Iteration "+str(ites)+"...")
        ites += 1
        if cost < 0.35 and flag  == 0:
            singular, check_real, check_imag = singu(N, U, alphas, check_real, check_imag, depth)
            flag = 1
        cost = loss_svd(alphas, N, depth, U)
        i_vec=[]
        for j in range(len(alphas)):
            i_vec.append(j)
        args = [( U, N, depth, alphas,i_vec[i]) for i in range(len(i_vec))]
        with multiprocessing.get_context('spawn').Pool(cores) as pool:
            g_dat=pool.map(task_wrapper, args)
            pool.close()
            pool.join()
        grad=g_dat
        for i in range(3 * N * depth):
            m1[i] = b_1 * m1[i] + (1 - b_1) * grad[i]
            v1[i] = b_2 * v1[i] + (1 - b_2) * grad[i] ** 2
            m_hat1[i] = m1[i] / (1 - b_1 ** (ites))
            v_hat1[i] = v1[i] / (1 - b_2 ** (ites))
            alphas[i] -= alpha * m_hat1[i] / (np.sqrt(v_hat1[i]) + epsilon)
    singular, check_real, check_imag = singu(N, U,alphas, check_real, check_imag, depth)
    final_circ  = QuantumCircuit(N)
    for y in range(depth):
        final_circ =  block_svd_def(final_circ, y, N, alphas)
    O = qi.Operator(final_circ)
    W = UnitaryGate(O)
    for i in range(len(singular)):
        singular[i] /= (2 * np.pi) #Normalization of the phase angles.
    return cost, singular, W
##############################################################

#Calling SVD(N, depth, U, iterations, cores)
##############################################################
if __name__ == "__main__":
    N = 1
    d = 2 ** N
    depth = 7
    iterations = 5
    cores = 1 #If set > 1, it allows for the parallellization of the gradient computation.
    U_Haar = random_unitary(d) #Haar-random target
    W_Haar = UnitaryGate(U_Haar)
    print("Starting...")
    cost, singular, W = SVD(N, depth, W_Haar, iterations, cores)
    V = W.to_matrix()
    np.savetxt('unitary.txt', V, fmt = '%.10f')
    np.savetxt('singulars.txt', singular, fmt = '%.10f')
##############################################################
