# Importing Libraries
from qiskit import QuantumCircuit, transpile
import qiskit.quantum_info as qi
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit.add_control import add_control
from qiskit.result import marginal_counts
import numpy as np
from random import randint
from qiskit_aer import Aer
from qiskit.circuit.library.generalized_gates import UnitaryGate

from qiskit.quantum_info import random_unitary
from multiprocessing import Pool

alpha = 0.001
b_1 = 0.9            
b_2 = 0.999
#m = np.zeros(size * (depth * 2 + 1))
#v = np.zeros(size * (depth * 2 + 1))
#m_hat = np.zeros(size * (depth * 2 + 1))
#v_hat = np.zeros(size * (depth * 2 + 1))

# Training hyperparameter
import qiskit
learning_rate = 0.1
noise_prob = 0.04 # [0, 1]
gamma = 0.7 # learning rate decay rate
delta = 0.01 # minimum change value of loss value
discounting_factor = 0.3 # [0, 1]
backend = Aer.get_backend('qasm_simulator')

# For parameter-shift rule
two_term_psr = {
    'r': 1/2,
    's': np.pi / 2
}

four_term_psr = {
    'alpha': np.pi / 2,
    'beta' : 3 * np.pi / 2,
    'd_plus' : (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}

one_qubit_gates = ["Hadamard", 'RX', 'RY', 'RZ']
two_qubits_gates = ['CNOT', 'CY', 'CZ', 'CRX', 'CRY', 'CRZ']

def create_gate_pool(num_qubits, one_qubit_gates = one_qubit_gates, two_qubits_gates = two_qubits_gates):
    gate_pool = []

    # Single-qubit gates
    single_qubit_gates = one_qubit_gates
    for qubit in range(num_qubits):
        for gate in single_qubit_gates:
            gate_pool.append((gate, qubit))

    # Two-qubit gates
    two_qubit_gates = two_qubits_gates
    for qubit1 in range(num_qubits):
        for qubit2 in range(num_qubits):
            if qubit1 != qubit2:
                for gate in two_qubit_gates:
                    gate_pool.append((gate, qubit1, qubit2))

    return gate_pool

# For QNG
generator = {
    'cu': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'crx': -1 / 2 * np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'ry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'cry': -1 / 2 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'rz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'crz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'cz': -1 / 2 * np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'i': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    'id': np.array([[1, 0], [0, 1]], dtype=np.complex128),
    '11': np.array([[0, 0], [0, 1]], dtype=np.complex128),
}

ignore_generator = [
    'barrier'
]
parameterized_generator = [
    'rx', 'ry', 'rz', 'crx', 'cry', 'crz'
]

# This information is extracted from http://dx.doi.org/10.1103/PhysRevA.83.042314
edges_graph_state = {
    2: ["0-1"],
    3: ["0-1", "0-2"],
    4: ["0-2", "1-3", "2-3"],
    5: ["0-3", "2-4", "1-4", "2-3"],
    6: ["0-4", "1-5", "2-3", "2-4", "3-5"],
    7: ["0-5", "2-3", "4-6", "1-6", "2-4", "3-5"],
    8: ["0-7", "1-6", "2-4", "3-5", "2-3", "4-6", "5-7"],
    9: ["0-8", "2-3", "4-6", "5-7", "1-7", "2-4", "3-5", "6-8"],
    10: ["0-9", "1-8", "2-3", "4-6", "5-7", "2-4", "3-5", "6-9", "7-8"]
}

look_up_operator = {
    "Identity": 'I',
    "Hadamard": 'H',
    "PauliX": 'X',
    'PauliY': 'Y',
    'PauliZ': 'Z',
    'S': 'S',
    'T': 'T',
    'SX': 'SX',
    'CNOT': 'CX',
    'CZ': 'CZ',
    'CY': 'CY',
    'SWAP': 'SWAP',
    'ISWAP': 'ISWAP',
    'CSWAP': 'CSWAP',
    'Toffoli': 'CCX',
    'RX': 'RX',
    'RY': 'RY',
    'RZ': 'RZ',
    'CRX': 'CRX',
    'CRY': 'CRY',
    'CRZ': 'CRZ',
    'U1': 'U1',
    'U2': 'U2',
    'U3': 'U3',
    'IsingXX': 'RXX',
    'IsingYY': 'RYY',
    'IsingZZ': 'RZZ',
}

#Task functions
def decimalToBinary(ip_val, size):
    a=bin(ip_val)
    b=str(a)
    
    if(ip_val<2):
        t4,t5,t6=b
        t5=0
        t4=0
        t3=0
        t1=0
        t2=0
        t0=0
        tc=0
        tb=0
        ta=0

    elif(ip_val<4):
        t3,t4,t5,t6=b
        t4=0
        t3=0
        t2=0
        t1=0
        t0=0
        tc=0
        tb=0
        ta=0

    elif(ip_val<8):
        t2,t3,t4,t5,t6=b
        t3=0
        t2=0
        t1=0
        t0=0
        tc=0
        tb=0
        ta=0

    elif(ip_val<16):
        t1,t2,t3,t4,t5,t6=b
        t2=0
        t1=0
        t0=0
        tc=0
        tb=0
        ta=0

    elif(ip_val<32):
        t0,t1,t2,t3,t4,t5,t6=b
        t1=0
        t0=0
        tc=0
        tb=0
        ta=0

    elif(ip_val<64):
        ta,t0,t1,t2,t3,t4,t5,t6=b
        t0=0
        tc=0
        tb=0
        ta=0
    elif(ip_val<128):
        tb,ta,t0,t1,t2,t3,t4,t5,t6=b
        tc=0
        tb=0
        ta=0
       

    else:
        tc,tb,ta,t0,t1,t2,t3,t4,t5,t6=b 
    if size == 1: 
        output = int(t6)
    elif size == 2:
        output = int(t5), int(t6)
    elif size == 3:
        output = int(t4), int(t5), int(t6)
    elif size == 4:
        output =  int(t3), int(t4), int(t5), int(t6)
    elif size == 5:
        output = int(2), int(t3), int(t4), int(t5), int(t6)
    elif size == 6:
        output = int(t1), int(t2), int(t3), int(t4), int(t5), int(t6)
    return output





def block_def(s, n, circuit, q, thetas):
    aux = 0
    for p in range(n): 
                circuit.rz(thetas[3*s*n+p*3], q[p])
                circuit.ry(thetas[3*s*n+p*3+1], q[p])
                circuit.rz(thetas[3*s*n+p*3+2 ], q[p])
    if s % 2 != 0:
        aux = 1
    for p in range(n):
        if 2 * p < n - (1 + aux):
            circuit.cx(q[2 * p + aux], q[2*p + 1 + aux]) 
        

 
                       
    return circuit

def cont_block(s, z, n, circuit, q, thetas):
    N = 1 #2 ** n
    aux = 0
    for p in range(n):
                circuit.crz(thetas[s][3*p], control_qubit = q[N * n + z] , target_qubit =  q[z*n + p])
                circuit.cry(thetas[s][3*p + 1 ],  control_qubit = q[N * n + z], target_qubit  = q[z*n +p ])
                circuit.crz(thetas[s][3*p+ 2],  control_qubit = q[N * n + z], target_qubit= q[z*n + p])
    
    if s % 2 != 0:
        aux = 1
    
    for p in range(n):
        if 2 * p < n - (1 + aux):
                        circuit.ccx(control_qubit1 = q[N * n + z], control_qubit2 = q[z*n + 2*p + aux], target_qubit =  q[z*n + 2*p + 1 + aux]) 
    return circuit




##################################################################################################################
#Derivations

def c_rot_derivative(circuit, i, j, thetas, n, X, states, cont_block,depth, F,q, c):
    y_1 = (np.sqrt(2) + 1)/(4*np.sqrt(2))
    y_2 = (np.sqrt(2) - 1)/(4*np.sqrt(2))
    diff = np.pi/2
    angles = np.copy(thetas)
    
    losses = []
    for p in range(2):
        angles[i][j] += diff * (1 + p)
        aux = circuit.copy()
        losses.append(F(aux, angles, n, X, states, cont_block, depth, q, c))
        angles[i][j] -= 2 * diff * ( 1 + 2 * p)
        aux = circuit.copy()
        losses.append(F(aux, angles, n, X, states, cont_block, depth, q, c))
        angles[i][j] += 2 * diff * ( 1 + 2 * p)
    der = y_1*(losses[0] - losses[1]) - y_2*(losses[2] - losses[3])
    return der



def grad_loss(qc: QuantumCircuit,aux_circuit,Mu, thetas, n, X, cont_block, depth, q, c,CW,i):

    count=0
    #print("Thetas:")
    #print(thetas)
    #This function tells you about the gates used and the type of phase shift rule to be used 2term or 4term if single or control gate
    #

    #index_list=np.zeros(len(thetas))
    #print(size)
    #grad_loss = np.zeros(len(thetas))#

    #print("Index:")
    #print(index_list)
    #print("Index_list")
    #print(index_list)#
    #print(index_list)


    grad_loss = single_4term_psr(qc, aux_circuit,Mu, thetas, n, X, cont_block, depth, q, c,CW,i)

    return grad_loss



def single_4term_psr(qc: QuantumCircuit,
                     aux_circuit,Mu, thetas, n, X,  cont_block, depth, q, c,CW, i):
    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas3, thetas4 = thetas.copy(), thetas.copy()
    thetas1[i] += four_term_psr['alpha']
    thetas2[i] -= four_term_psr['alpha']
    thetas3[i] += four_term_psr['beta']
    thetas4[i] -= four_term_psr['beta']

    #Pass these changes through the completed circuit 

    qc1=[]
    qc2=[]
    qc3=[]
    qc4=[]


    #Create a for cycle for each initialization on each of these
    #Save the circuits in appending vectors
    #Try to paralellize this then

    for yi in range(4**n):

        qc1.append(create_whole_circuit(aux_circuit,Mu, thetas1, n, X, cont_block, depth, q, c,CW,yi))
        qc2.append(create_whole_circuit(aux_circuit,Mu, thetas2, n, X, cont_block, depth, q, c,CW,yi))
        qc3.append(create_whole_circuit(aux_circuit,Mu, thetas3, n, X, cont_block, depth, q, c,CW,yi))
        qc4.append(create_whole_circuit(aux_circuit,Mu, thetas4, n, X, cont_block, depth, q, c,CW,yi))


    A=sum_frequencies_all(qc1,n)
    B=sum_frequencies_all(qc2,n)
    C=sum_frequencies_all(qc3,n)
    D=sum_frequencies_all(qc4,n)

#   sum_frequencies(qc1, n)



    return - (four_term_psr['d_plus'] * (
        A -
        B) - four_term_psr['d_minus'] * (
        C -
        D))



#sum_frequencies(qc1, n)

#############################################################################################################
def sum_frequencies_all2(vec_circuit, n):
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 1000
    N = 1#2 ** n
    lista = []
    #print("pass2")
    for yi in range(4**n):
        circ=vec_circuit[yi]
        qpe2 = circ.copy()
        qpe2.measure([N *n], [0])
                                                               
        T_aux_circuit = transpile(qpe2, aer_sim)
        results = aer_sim.run(T_aux_circuit, shots=shots).result()
        frequency = marginal_counts(results, indices=[0]).get_counts()['1']/shots
                                                                                       
        lista.append(frequency)

    suma  = sum(lista)/(4**n)
                                                                                                                                                               
    return suma

def sum_frequencies_all(vec_circuit, n):
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 1000 
    N = 1#2 ** n
    lista = []
    for yi in range(4**n):
        circ=vec_circuit[yi]
        qpe2 = circ.copy()
        qpe2.measure([N *n], [0])
        T_aux_circuit = transpile(qpe2, aer_sim)
        results = aer_sim.run(T_aux_circuit, shots=shots).result()
        frequency = marginal_counts(results, indices=[0]).get_counts()['1']/shots
        lista.append(frequency)
    suma  = sum(lista)/(4**n)
    return suma


def cohe_sum_1(circuit, n, q, CW,  thetas, depth):
    circuit.h(n)
    a=np.arange(0,n)
    At=a.tolist()
    Bt=[n]
    T=np.concatenate((Bt, At), axis=0)
    lista=T.tolist()
    circuit.append(CW, lista)

    return circuit

def cohe_sum_2(circuit, n, q, cont_block, thetas, depth):
    N = 1#2 ** n
    for k in range(N):
        circuit.x(N * n + k)
        for m in range(depth): 
            circuit = cont_block(m, k, n, circuit, q, thetas )
        for i in range(n):
            circuit.crz(thetas[depth][3*i], control_qubit = q[N * n + k] , target_qubit =q[k*n + i])
            circuit.cry(thetas[depth][3*i + 1 ],control_qubit = q[N * n + k] , target_qubit = q[ k*n + i])
            circuit.crz(thetas[depth][3*i + 2], control_qubit = q[N * n + k] , target_qubit =q[k*n + i])
        circuit.x(N * n + k)
        circuit.h(N * n + k)
    return circuit

def sum_frequencies(circuit, n):
    aer_sim = Aer.get_backend('aer_simulator')
    shots = 1000 
    N = 1#2 ** n
    lista = []
    for k in range(N):
        qpe2 = circuit.copy()
        qpe2.measure([N *n + k], [k])
        T_aux_circuit = transpile(qpe2, aer_sim)
        results = aer_sim.run(T_aux_circuit, shots=shots).result()
        try: 
            frequency = marginal_counts(results, indices=[k]).get_counts()['0']/shots
        except:
            frequency = 0
            
        else:
            frequency = marginal_counts(results, indices=[k]).get_counts()['0']/shots
        
        lista.append(frequency)
    #print(qpe2)
        
    suma  = sum(lista)/N
    #print(qpe2)
    
    return suma

def init(circuit, n, q, thetas,yi):
    N = 1
    
    a=np.arange(0,n)
    At=a.tolist()
    vec = random_statevector(2**n,seed=yi)
    initial_state = circuit.initialize(vec.data, At)
    return circuit
    
def target_prep(thetas, n, X, cont_block, depth,CW,yi= 0):

    q_cost = QuantumRegister(n + 1, 'q_cost')
    c_cost = ClassicalRegister(1, 'c_cost')
    cost_circuit = QuantumCircuit(q_cost, c_cost)
    cost_circuit = init(cost_circuit, n, q_cost, thetas,yi)
    cost_circuit = cohe_sum_1(cost_circuit, n, q_cost, CW, thetas, depth)
    
    return cost_circuit, q_cost, c_cost


def cost(cost_circuit, thetas, n, X, cont_block, depth, q_cost, c_cost):
    N = 1#2 ** n
    value = sum_frequencies(cost_circuit, n)
    return value/N

def Create_VQC_circ(q2,U_Gate, thetas, n, X, cont_block, depth, q_cost, c_cost):
    N = 1#2 ** n

    CH = add_control(U_Gate,1, ctrl_state=0,label="U_L")

    a=np.arange(0,n)
    At=a.tolist()
    Bt=[n]
    T=np.concatenate((Bt, At), axis=0)
    Tot=T.tolist()
    q2.append(CH, Tot)

    for j in range(N):
        q2.h(N * n + j)

    return q2

def create_whole_circuit(Aux,U_Gate, thetas, n, X, cont_block, depth, q_cost, c_cost,CW,yi=0):

    q_def = QuantumRegister(n, 'q_def')
    circ2 = QuantumCircuit(q_def)

        #Op:fast 0.006s
    for m in range(depth): 
        circ2 = block_def(m, n, circ2, q_def, thetas)
        #Op:fast 0.005s
    for i in range(n):
        circ2.rz(thetas[3*depth*n+i*3], q_def[i])
        circ2.ry(thetas[3*depth*n+i*3+1 ], q_def[i])
        circ2.rz(thetas[3*depth*n+i*3+2], q_def[i])
        
    M = qi.Operator(circ2)
    U_Gate=UnitaryGate(M.data,label="Learn_U")
    q2,q,c = target_prep(thetas, n, X, cont_block, depth,CW,yi)
    q2=Create_VQC_circ(q2,U_Gate, thetas, n, X, cont_block, depth, q_cost, c_cost) # Create VQC and attach to q2

    return q2



def Grad_calc_bone(ref_circuit,i,j,thetas, n, X, cont_block,depth,  cost, q,c,N):
    grad = c_rot_derivative(ref_circuit, i, j, thetas, n, X, cont_block,depth,  cost, q,c) #/N
             

    return grad

def ADAM_tomo(X, n, ites, depth, cont_block, cost,cores):
    N = 1#2 ** n
    alpha = 0.1
    b_1 = 0.8            
    b_2 = 0.999
    epsilon = 10 ** (-8)
    w = np.zeros(depth*n*3+3*n)
    v = np.zeros(depth*n*3+3*n)
    w_hat = np.zeros(depth*n*3+3*n)
    v_hat = np.zeros(depth*n*3+3*n)
    thetas = np.ones(depth*n*3+3*n)


    #Op: fast
    #for i in range(depth*n*3+3*n):
    #    thetas[i]= np.random.uniform(0, 2 * np.pi)
    W = UnitaryGate(X)
    CW = add_control(W, ctrl_state='1',label="CW", num_ctrl_qubits = 1) # this could be passed down because this does not change, could be done generally

    #Op: pain
    qk=[]
    for t in range(ites):
        runtime = time()

        ref_circuit,q,c = target_prep(thetas, n, X, cont_block, depth,CW)    
        aux_circuit = ref_circuit.copy()

        ##########################################################################################################
        #Done just to test block, what is the unitary
        q_def = QuantumRegister(n, 'q_def')
        circ = QuantumCircuit(q_def)

        #Op:fast 0.006s
        for m in range(depth): 
            circ = block_def(m, n, circ, q_def, thetas)

        #Op:fast 0.005s
        for i in range(n):
            circ.rz(thetas[3*depth*n+i*3], q_def[i])
            circ.ry(thetas[3*depth*n+i*3+1 ], q_def[i])
            circ.rz(thetas[3*depth*n+i*3+2], q_def[i])
        print("Variational circuit ------------------------------------------------------------------")
        print("........................")
        print(t) 
        M = qi.Operator(circ)
        Mu=UnitaryGate(M.data,label="Learn_U")
        ##########################################################################################################
        print(X+M)
      
        q2=create_whole_circuit(aux_circuit,Mu, thetas, n, X, cont_block, depth, q, c,CW)

        i_vec=[]

        for j in range(len(thetas)):
            i_vec.append(j)

        args = [(q2,aux_circuit,Mu, thetas, n, X, cont_block, depth, q, c,CW,i_vec[i]) for i in range(len(i_vec))]

        #Input for gradient should be the full ref_circuit, not the weird combination ------------------------- so you should restructure the whole segment
        with Pool(cores) as pool:
        # Pool map for the seed in paralel.
            g_dat=pool.map(task_wrapper, args)
            pool.close()
            # wait for all issued tasks to complete
            pool.join()
        
        grad=g_dat
    
        #Op: fast depth=1 0.005s
        for i in range(len(thetas)):
            w[i] = b_1 * w[i] + (1 - b_1) * grad[i]
            v[i] = b_2 * v[i] + (1 - b_2) * grad[i] ** 2
            w_hat[i] = w[i] / (1 - b_1 ** (t + 1))
            v_hat[i] = v[i] / (1 - b_2 ** (t + 1))
            thetas[i] -= alpha * w_hat[i] / (np.sqrt(v_hat[i]) + epsilon)

        runtime2 = time() - runtime

        j_vec=[]
        for j in range(4**n):
            j_vec.append(j)


        args2 = [(aux_circuit,Mu, thetas, n, X, cont_block, depth, q, c,CW,j_vec[i]) for i in range(len(j_vec))]

        #Input for gradient should be the full ref_circuit, not the weird combination ------------------------- so you should restructure the whole segment
        with Pool(cores) as pool:
        # Pool map for the seed in paralel.
            all_cir=pool.map(task_wrapper2, args2)

        
            pool.close()
            # wait for all issued tasks to complete
            pool.join()

        qk=qk+all_cir

    return M, U,runtime2,qk



# wrapper function for task
def task_wrapper(args):
    # call task() and unpack the arguments
    #Use your function that you want to parallleize
    return grad_loss(*args)


def task_wrapper2(args):
    # call task() and unpack the arguments
    #Use your function that you want to parallleize
    return create_whole_circuit(*args)


from qiskit.quantum_info import random_statevector
from multiprocessing.pool import ThreadPool
from time import time
from qiskit.quantum_info import random_unitary
import numpy as np

if __name__ == "__main__":

    cores=1
    seedU = 42
    n = 2
    d = 2 ** n
    I = np.zeros((d, d))
    ites = 2
    depth = 1
    for i in range(d):
        I[i][i] = 1

    q_cost = QuantumRegister(n, 'q_cost')
    c_cost = ClassicalRegister(n, 'c_cost')
    priorD = QuantumCircuit(q_cost, c_cost)
    

    U = random_unitary(d, seed=seedU)

    Learn_U,_,runtime2,qk=ADAM_tomo(U, n, ites, depth, cont_block, cost,cores)
    costt=[]
    for j in range(ites-1):
        costt.append(sum_frequencies_all2(qk[j*(4**n):(j+1)*(4**n)],n))
    
    print("Starting h o h")
    print("Finished depth: "+str(depth)+ ", ites"+str(ites)+", cores: "+str(cores))
    print("Runtime: "+str(runtime2))
    print(" ")
    print("Cost"+str(costt))
    print(np.array(Learn_U))
    print(" ")
    print(np.array(U))
    print("-----------------------------------------------------------------------------")
