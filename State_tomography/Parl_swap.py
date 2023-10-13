
# Importing standard Qiskit libraries
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit_aer import AerSimulator
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from random import randint
# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
import numpy as np
#More needed packages
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.circuit.library.standard_gates import HGate
import numpy as np
import matplotlib.pyplot as plt
from time import time
import random
from time import time

import compare_W
#Block 

#Seeds definition
seedU=42
seedT = 42

depth = 1      #Layer depth
size = 3       #Qubit size of psi
ites = 2      #Epochs
Max_Tr=1       #Number of trials
num_shots=10000

#Initilization of hyperparameters for ADAM oprimizer

alpha = 0.001
b_1 = 0.9            
b_2 = 0.999
epsilon = 10 ** (-8)
m = np.zeros(size * (depth * 2 + 1))
v = np.zeros(size * (depth * 2 + 1))
m_hat = np.zeros(size * (depth * 2 + 1))
v_hat = np.zeros(size * (depth * 2 + 1))

# Training hyperparameter
import qiskit
learning_rate = 0.1
noise_prob = 0.04 # [0, 1]
gamma = 0.7 # learning rate decay rate
delta = 0.01 # minimum change value of loss value
discounting_factor = 0.3 # [0, 1]
backend = qiskit.Aer.get_backend('qasm_simulator')

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



def create_block2(qc: qiskit.QuantumCircuit, thetas: np.ndarray,thetas2: np.ndarray,thetas3: np.ndarray,size:float):

	#print(thetas,thetas2,thetas3)

    count=0
    for j in range(1+size,1+size*2):#
        qc.rx(thetas[count], j)#
        count=count+1
    #qc.barrier()
    for j in range(1+size,size*2,2):
        qc.cnot(j, j+1) 
    count=0
    #qc.barrier()
    for j in range(1+size,1+size*2):
        qc.ry(thetas2[count], j)
        count=count+1
    #qc.barrier()
    for j in range(1+size,size*2-1,2):
        qc.cnot(j+1, j+2)
    count=0
    for j in range(1+size,1+size*2):
        qc.rx(thetas3[count], j)
        count=count+1
    qc.barrier()



    #count=0
    #for j in range(1+size,1+size*2):#
        #print(j,size)
    #    if (j+1==1+size*2):
    #        qc.cry(thetas[count], j,1+size)#

    #    else:
            #print(j,j+1)
    #        qc.cry(thetas[count], j,j+1)#
    #    count=count+1
    #qc.barrier()
    #count=0
    #for j in range(1+size,1+size*2):
    #    qc.rx(thetas2[count], j)    
    #    qc.ry(thetas3[count], j) 
    #    count=count+1
    #qc.barrier()

    return qc

#### This is variational circuit definition using the cx gates
def create__ansatz2(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,size:float,
                               num_layers: int = 1):

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #print(thetas)
    n = size

    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas
        #print(thetas)
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_block2(qc, phis[n*3*i:n*3*i+n], phis[n*3*i+n:n*3*i+2*n],phis[n*3*i+2*n:n*3*i+3*n],size)
        
        #qc.barrier()
        #qc = create_rz_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])
        #print(qc)

    #thetas3=phis[len(thetas)-n:len(thetas)]

    #Last layer is a rx
    #count=0
    #for j in range(1+size,1+size*2):
    #    qc.rx(thetas3[count], j)
    #    count=count+1
    #qc.barrier()
    #print(qc)
    return qc


from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeHanoi
from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit.providers.fake_provider import FakeWashington
from qiskit.providers.fake_provider import FakeCambridge
from qiskit.providers.aer import QasmSimulator, AerSimulator

def split_circuit(circuit, start, end):
    nq = len(circuit.qubits)
    qc2 = qiskit.QuantumCircuit(nq)
    for x in circuit[start:end]:
        qc2.append(x[0], x[1])
    return qc2


def Var_circ(size, depth,thetas):
    #Combine the swap test with the variational algorithm and cnots #################################### and create a repeatable block
    q = QuantumRegister(1 + 4 * size,'q')   
    c = ClassicalRegister(depth,'c')
    qpe2 = QuantumCircuit(q, c)
    qpe2 = create__ansatz2(qpe2, thetas, size, 1) 

    for k in range(size): #create a cnot for U to aux and V to aux
        qpe2.cnot(q[1+k],q[1+k+2*size])
        qpe2.cnot(q[1+k+size],q[1+k+3*size])


    qpe2.h(0)
    for i in range(size):
        qpe2.cswap(q[0], q[1+i+2*size],q[1+i+3*size])

    qpe2.h(0)

    return qpe2


#Code to define the constant update of the thetas
def circuit_update2(thetas, depth, size):

	#U_Haar = random_unitary(2**size, seed=seedU)#
    U_Haar=compare_W.Create_Haar_U(size,42)
	#To perform a comparison with the long method

    CH=UnitaryGate(U_Haar.data,label="Har")
    
    q = QuantumRegister(1 + 4 * size,'q')	
    c = ClassicalRegister(depth,'c')
    qc = QuantumCircuit(q, c)
    #Define Haar target 
    lista = []
	##for i in range(size):
	##	lista.append(q[1+i])
    #cost_circuit.unitary(U_Haar, lista)
    #Define GHZ target
	##qc.h(1)
	
	##for y in range(size-1):
	##	qc.cnot(1, 2+y)
    a=np.arange(1,size+1)
    At=a.tolist()
    qc.append(CH, At)
    
    qc1=qc.decompose()

    D_2=qc1.depth()
    D_2=int(D_2/depth) #Normalized size of the circuit (blocks)

    
    ###########################################################################################

    C=[]
    #Create N=depth smaller circuits to perform the simulation
    for bq in range(depth):
        c1=split_circuit(qc1,D_2*bq,D_2*bq+D_2) #Creates blocks of the circuit uniformely separated
        #Now for first circuit create a swap test, with variational algorithm initialization then save the circuit, perform this for all blocks
        #In the end you will use circuit_compose to combine all the circuits all together

        c1_comp=c1.compose(Var_circ(size, depth,thetas[bq])) # Here you have the hole block

        C.append(c1_comp)
        

    #print("----------------------------------------------------------------------------")
    #print(C[0])
    Aux=np.arange(1+size,1+4*size)
    Aux2=Aux.tolist()
    q = QuantumRegister(1 + 4 * size,'q')   
    c = ClassicalRegister(depth,'c')
    qpe3 = QuantumCircuit(q, c)
    
    for k in range(len(C)):
        qpe3=qpe3.compose(C[k])
        qpe3.measure([0],[k])
        qpe3.reset([0])
        qpe3.reset(Aux2)
        for l in range(size):
            qpe3.cnot(1+l,1+size+l)

    #aer_sim = FakeGuadalupe()
    #tqc = transpile(qc1, aer_sim,seed_transpiler=seedT)
    #print(qpe3)
    #print(tqc.decompose())
    ########################################################### Step: 1 create a circuit ansatz for middle of the circuit 
    

    #####################################################

    #print(C[1])
    
    # Create a transpilation of the circuit so I can save it for later use
   # swap_test_circuit = qc.copy()
   # swap_test_circuit.h(0)
   # for i in range(size):
   #     swap_test_circuit.cswap(q[0], q[i+1],q[1+ size + i])

   # swap_test_circuit.h(0)
    #swap_test_circuit.measure(q[0], c[0])
    
    return qpe3


def adam2(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, iteration: int,
         grad_loss2: np.ndarray,size,depth):
    """Adam2 Optimizer. Below codes are copied from somewhere :)

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam2
        - v (np.ndarray): params for Adam2
        - i (int): params for Adam2
        - grad_loss2 (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    #num_thetas = thetas.shape[0]
    beta1, beta2, epsilon = 0.8, 0.999, 10**(-8)
    for j in range(depth):
        for i in range(0, 3*size):
            m[j][i] = beta1 * m[j][i] + (1 - beta1) * grad_loss2[j][i]
            v[j][i] = beta2 * v[j][i] + (1 - beta2) * grad_loss2[j][i]**2
            mhat = m[j][i] / (1 - beta1**(iteration + 1))
            vhat = v[j][i] / (1 - beta2**(iteration + 1))
            thetas[j][i] -= learning_rate * mhat / (np.sqrt(vhat) +
                                                          epsilon)

    return thetas

def measure_i(qc: QuantumCircuit,j):
    T_swap_test_circuit = transpile(qc, backend,seed_transpiler=seedT)
	#shots = 1000
    results = backend.run(T_swap_test_circuit, shots=num_shots).result() #Obtain marginal shots and try to minimize both gradient based on this,

    #Add marginal and update for j in parallel,

    freq1 = results.get_counts()['0']/num_shots
   # freq2 = marginal_counts(results, indices=[1]).get_counts()['0']/num_shots

    #frequency = results.get_counts()['0'] / num_shots
	#fide = np.sqrt(1 - 2 * frequency) #renormalization of the value of fidelity
	#print("Frequency of appearing a 0:")
	#print(frequency)
    return freq1


def loss_func(measurement):


    return np.sqrt(1-measurement)


def get_wires_of_gate(gate):
    """Get index bit that gate act on

    Args:
        - gate (qiskit.QuantumGate): Quantum gate

    Returns:
        - numpy arrray: list of index bits
    """
    list_wire = []
    for register in gate[1]:
        list_wire.append(register.index)
    return list_wire


def is_gate_in_list_wires(gate, wires):
    """Check if a gate lies on the next layer or not

    Args:
        - gate (qiskit.QuantumGate): Quantum gate
        - wires (numpy arrray): list of index bits

    Returns:
        - Bool
    """
    list_wire = get_wires_of_gate(gate)
    for wire in list_wire:
        if wire in wires:
            return True
    return False

def split_into_layers(qc: qiskit.QuantumCircuit):
    """Split a quantum circuit into layers

    Args:
        - qc (qiskit.QuantumCircuit): origin circuit

    Returns:
        - list: list of list of quantum gates
    """
    layers = []
    layer = []
    wires = []
    is_param_layer = None
    for gate in qc.data:
        name = gate[0].name
        if name in ignore_generator:
            continue
        param = gate[0].params
        wire = get_wires_of_gate(gate)
        if is_param_layer is None:
            if len(param) == 0:
                is_param_layer = False
            else:
                is_param_layer = True
        # New layer's condition: depth increase or convert from non-parameterized layer to parameterized layer or vice versa
        if is_gate_in_list_wires(gate, wires) or (is_param_layer == False and len(param) != 0) or (is_param_layer == True and len(param) == 0):
            if is_param_layer == False:
                # First field is 'Is parameterized layer or not?'
                layers.append((False, layer))
            else:
                layers.append((True, layer))
            layer = []
            wires = []
        # Update sub-layer status
        if len(param) == 0 or name == 'state_preparation_dg':
            is_param_layer = False
        else:
            is_param_layer = True
        for w in wire:
            wires.append(w)
        layer.append((name, param, wire))
    # Last sub-layer
    if is_param_layer == False:
        # First field is 'Is parameterized layer or not?'
        layers.append((False, layer))
    else:
        layers.append((True, layer))
    return layers


def get_cry_index(thetas: np.ndarray,depth,size):
    """Return a list where i_th = 1 mean thetas[i] is parameter of CRY gate

    Args:
        - func (types.FunctionType): The creating circuit function
        - thetas (np.ndarray): Parameters
    Returns:
        - np.ndarray: The index list has length equal with number of parameters
    """
    qc = circuit_update2(thetas,depth,size)		

    layers = split_into_layers(qc)
    index_list = []
    for layer in layers:
        for gate in layer[1]:
            if gate[0] == 'cry':
                index_list.append(1)
            else:
                index_list.append(0)
            if len(index_list) == len(thetas):
                return index_list
    #print("Cry for circuit:")
    #print(qc)
    #print(" ")
    return index_list

import math
from qiskit.result import marginal_counts


def grad_loss2(qc: QuantumCircuit,
              thetas,size,depth):
    
    #print(thetas)
    count=0

    #### Create a for cycle with fractions of theta that are being filled
    #grad_counter=




    #This function tells you about the gates used and the type of phase shift rule to be used 2term or 4term if single or control gate
    index_list=[[0]*3*size for i in range(depth)]

    #print(size)
    s=[[0]*3*size for i in range(depth)]
    grad_loss2 = [[0]*3*size for i in range(depth)]
    depth_counter=0
    counter=0
    for gate in qc.data:#
        #print(qc.data)
		#print(gate[0].name,gate[0].params)#if(str(gate[1])=="[Qubit(QuantumRegister("+str(1+size*2)+", 'q'), "+str(size+1)+")]"or str(gate[1])=="[Qubit(QuantumRegister("+str(1+size*2)+", 'q'), "+str(size+2)+")]" or str(gate[1])=="[Qubit(QuantumRegister("+str(1+size*2)+", 'q'), "+str(size+3)+")]"):
        depth_counter_n=math.trunc(counter/(3*size))
        #print(depth_counter_n,counter)
        
        if (depth_counter!=int(depth_counter_n)):
            depth_counter=depth_counter_n
            count=0
        #print("-------------------------------")
        #print(depth_counter,count/(3*size),index_list)
        #print("_-----------------------------")

        #print(index_list)
        if(gate[0].name=="rx" or gate[0].name=="ry" or gate[0].name=="rz"):
            #print(depth_counter,count)
            index_list[depth_counter][count]=0
            count=count+1
            counter=counter+1

        elif(gate[0].name=="crx" or gate[0].name=="cry" or gate[0].name=="crz"):
            #print(depth_counter,count)
            index_list[depth_counter][count]=1
            count=count+1
            counter=counter+1
   ## index_list = get_cry_index(thetas,depth,size)
	#print("Index_list")
	#print(index_list)
    for j in range(depth):
        for i in range(0, 3*size):
            if index_list[j][i] == 0:
                grad_loss2[j][i] = single_2term_psr2(qc, thetas,size,depth,
                               i,j)
            if index_list[j][i] == 1:
                grad_loss2[j][i] = single_4term_psr2(qc, thetas,size,depth,
                               i,j)
    return grad_loss2

def single_2term_psr2(qc: QuantumCircuit,
                     thetas,size,depth, i,j):
    thetas1, thetas2 = thetas.copy(), thetas.copy()


    thetas1[j][i] += two_term_psr['s']
    thetas2[j][i] -= two_term_psr['s']
    

    qc1 = circuit_update2(thetas1, depth, size) #+ c1
    qc2 = circuit_update2(thetas2, depth, size) #- c1

    ###########################################################################################
   # print("Measurement,"+str(i))
   # print(measure_i(qc1,j),measure_i(qc2,j))

    return -two_term_psr['r'] * (
        measure_i(qc1,j) -
        measure_i(qc2,j))

def single_4term_psr2(qc: QuantumCircuit,
                     thetas,size,depth, i,j):

    #Every j is independent from each other

    thetas1, thetas2 = thetas.copy(), thetas.copy()
    thetas3, thetas4 = thetas.copy(), thetas.copy()
    thetas1[j][i] += four_term_psr['alpha']
    thetas2[j][i] -= four_term_psr['alpha']
    thetas3[j][i] += four_term_psr['beta']
    thetas4[j][i] -= four_term_psr['beta']

    qc1 = circuit_update2(thetas1, depth, size)
    qc2 = circuit_update2(thetas2, depth, size)
    qc3 = circuit_update2(thetas3, depth, size)
    qc4 = circuit_update2(thetas4, depth, size)


    L1_loss=- (four_term_psr['d_plus'] * (
        measure_i(qc1,j) -
        measure_i(qc2,j)) - four_term_psr['d_minus'] * (
        measure_i(qc3,j) -
        measure_i(qc4,j)))



    return L1_loss

def Adam2_QST(depth, size, ites):
    
    s=[[0]*3*size for i in range(depth)]
    v = [[0]*3*size for i in range(depth)]
    s_hat = [[0]*3*size for i in range(depth)]
    v_hat = [[0]*3*size for i in range(depth)]
    thetas = [[0]*3*size for i in range(depth)] #Parameters to be trained

    loss = [0]*depth#Parameters to be trained

    for i in range(3*size):
        thetas[0][i]= np.random.uniform(0, 2 * np.pi)

    loss_values=[[],[]] #values for the loss
    
    #print("Thetas init -----------------------------------------------")
    #print(thetas)
    
    #Initialization
    #for i in range(depth*size*3):
    #    thetas[i]= np.random.uniform(0, 2 * np.pi)
        
    #print(size,depth)
    vdagger = circuit_update2(thetas, depth, size) #Create the first structure of the quantum circuit
    print(vdagger)
    for i in range(ites):
        runtime = time()
		
		#fide = -(factor)**(-1)/2
        
        #print(fide)
        #factors = np.zeros(depth*size*3) 
        #grad = np.zeros(depth*size*3)
        grad = grad_loss2(                ############################# we are here, input thetas,size,depth,circuit
            vdagger, thetas,size,depth)

        thetas=adam2(thetas,s,v,ites,grad,size,depth)
        
        #create an updated circuit with thetas
        v_copy = circuit_update2(thetas,depth,size)
        #print(v_copy)
		#print(v_copy)
        loss = loss_func(
            measure_i(v_copy,0))

        loss_values.append(loss)
            


        runtime2 = time() - runtime
        print("Step " + str(i) + ": " + str(loss)+"| Time: "+str(runtime2))
                                                                              
                            
    
    return loss_values
                                                                              
                            

def plot_loss(loss,Loss_cry,Loss_cx,depth,size,ites):


    plt.plot(loss,label='Cost_Loss_cx_swap')
    plt.plot(Loss_cry,label='Cost_Loss_cry_Vd')
    plt.plot(Loss_cx,label='Cost_Loss_cx_Vd')

	#plt.savefig('C:/Users/vladl/Documents/PHD/Simulations/Qiskit-Simulations/SECURITY/Modified_Pol/'+'CX'+str(depth)+"_"+str(size)+"_"+str(ites)+'.png', bbox_inches='tight') #If you want to save the circuit
    plt.legend()
    plt.show()

def plot_loss_simple(loss,depth,size,ites):


    plt.plot(loss,label='Cost_Loss_cx_swap')

    #plt.savefig('C:/Users/vladl/Documents/PHD/Simulations/Qiskit-Simulations/SECURITY/Modified_Pol/'+'CX'+str(depth)+"_"+str(size)+"_"+str(ites)+'.png', bbox_inches='tight') #If you want to save the circuit
    plt.legend()
    plt.show()


