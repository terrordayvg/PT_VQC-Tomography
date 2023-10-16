from random import random

#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions import *

# import basic plot tools
from qiskit.visualization import plot_histogram
# Let's see how it looks:
from numpy.random import uniform
from numpy import pi
import numpy as np
from scipy.signal import argrelextrema

from qiskit.quantum_info import random_unitary

from numpy.linalg import qr
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from multiprocessing import Pool


from qiskit.providers.fake_provider import FakeCambridge
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.quantum_info import state_fidelity

from qiskit.result import marginal_counts
import math

from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit_aer.noise import NoiseModel

import qutip as qt
from multiprocessing.pool import ThreadPool
from qiskit.quantum_info import random_statevector
from qiskit.circuit.add_control import add_control
from qiskit.quantum_info import Statevector


from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import QasmSimulator
import collections


def decimalToBinary(ip_val):
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



    return int(t6)

def Verification_block_batch(Iancilla,Phi1,Psi_dim,qpe2,ancilla_N):
    
    _dshottest=10000
    Pos_c1=1
    Rc_block(Iancilla,Phi1,Psi_dim,qpe2,ancilla_N)
    qpe2.h(Iancilla)
    qpe2.measure([Iancilla],[0])

    sim_toro = Aer.get_backend('aer_simulator')
    t_qpe2 = transpile(qpe2, sim_toro,seed_transpiler=42)
    qobj = assemble(t_qpe2, shots=_dshottest)
    results = sim_toro.run(qobj).result()
    finalm = results.get_counts()
    simp_counts2 = marginal_counts(results, indices=[0]).get_counts()
    #print(simp_counts2["0"])

    print(simp_counts2)
    if(_dshottest*0.9<simp_counts2["0"]):
        Pos_c1=0



    return qpe2,Pos_c1


def Verification_block(Iancilla,Phi1,Psi_dim,qpe2,ancilla_N):
    
    _dshottest=1
    Pos_c1=1
    Rc_block(Iancilla,Phi1,Psi_dim,qpe2,ancilla_N)
    qpe2.h(Iancilla)
    qpe2.measure([Iancilla],[0])

    sim_toro = Aer.get_backend('aer_simulator')
    t_qpe2 = transpile(qpe2, sim_toro,seed_transpiler=42)
    qobj = assemble(t_qpe2, shots=_dshottest)
    results = sim_toro.run(qobj).result()
    finalm = results.get_counts()
    simp_counts2 = marginal_counts(results, indices=[0]).get_counts()
    #print(simp_counts2["0"])

    print(simp_counts2)
    for i in range(2):
        num=decimalToBinary(i)
        ext=str(num)
    
        try:
          
            if (simp_counts2[ext] != 0):
                Pos_c1=i

        except:
            continue



    return qpe2,Pos_c1


#Create function of block W made of Rc_block(phi) H Rc_block(phi)
def W_block(Phi1,Phi2,Psi_dim,Iancilla,qpe2,ancilla_N):
    Rc_block(Iancilla,Phi1,Psi_dim,qpe2,ancilla_N)
    qpe2.h(Iancilla)
    Rc_block(Iancilla,Phi2,Psi_dim,qpe2,ancilla_N)

    return qpe2 

# Create function of block Rc
def Rc_block(anc_q,Phi,Psi_dim,qpe2,ancilla_N):
    #From qutip import a v v+ notation for |v><v|
    Phi_y=np.array(Phi)
    Phi_q=qt.Qobj(Phi_y)
    A = Phi_q * Phi_q.dag()
    I=qt.qeye(2**Psi_dim)


    I_d=I.full()
    A_d=A.full()


    # Tayler series expansion turns into this: e^{i*pi*|Phi><Phi)}=I-|Phi><Phi|+e^{i*pi}|Phi><Phi|
    Ex=I_d-2*A_d


    #Implement the controlled version of Ex, and turn in into a gate
    Rc=UnitaryGate(Ex.data,label="Rc")
    Rc_c = add_control(Rc,1, ctrl_state=1,label="Control")

    a=[anc_q]
    b=np.arange(ancilla_N,ancilla_N+Psi_dim, dtype=int) #Generalize vectors [counting_qubit]+[ancilla, ancilla+target]
    Vel=np.concatenate((a, b), axis=None)
    Vel=Vel.tolist() #Turn into a list to be an input for append


    qpe2.append(Rc_c, Vel)
    return qpe2

def Dataset1_qiskit(ancilla_N,Psi_dim,n_shots):

    #A=np.arange(n_shots)
    vec=[0]*n_shots
    vec2=[0]*n_shots
    on=[0]*Psi_dim
    #Version with quantum circuit and you obtain the measurements
    for j in range(n_shots):

        q = QuantumRegister(Psi_dim, 'q')
        c = ClassicalRegister(Psi_dim, 'c')
        qpe2 = QuantumCircuit(q,c)

        for k in range(Psi_dim):
            on[k]=np.random.uniform(0, pi)
            qpe2.ry(on[k],k)

        vec[j] = Statevector.from_instruction(qpe2)

        #Create a unitary database |Psi_in>, |Psi_out>
        ## Definition of the Heisenber 3d model
        CHar=random_unitary(2**Psi_dim, seed=42)
        CHar2=UnitaryGate(CHar.data,label="Har")

        a=np.arange(Psi_dim)
        Al=a.tolist()

        qpe2.append(CHar2, Al)

        backend = Aer.get_backend("statevector_simulator")
        result = execute(qpe2, backend=backend, shots=1,seed_transpiler=42).result()
        stateOut=result.get_statevector()        
        vec2[j]=stateOut
        #print(vec3[j])
        #print("-------------")

    return vec,vec2


def Dataset_statevector(ancilla_N,Psi_dim,n_shots):

    #A=np.arange(n_shots)


    v_basis=[0]*(2**Psi_dim)
    vec=[0]*(2**Psi_dim)
    vec2=[0]*(2**Psi_dim)
    vec3=[0]*n_shots


    #Create orthogonal basis, [0,0,0,1],[0,0,1,0]...=vec1
    #Evolve them according the U|Phi>orh=vec2
    for l in range(2**Psi_dim):
        vec[l]=Statevector(np.array(qp.basis(2**Psi_dim,l)))
        q = QuantumRegister(Psi_dim, 'q')
        c = ClassicalRegister(Psi_dim, 'c')
        qpe2 = QuantumCircuit(q,c)

        #vec[j]=random_statevector(2**Psi_dim,seed=j)
        #Create a unitary database |Psi_in>, |Psi_out>
        ## Definition of the Heisenber 3d model
        CHar=random_unitary(2**Psi_dim, seed=42)
        CHar2=UnitaryGate(CHar.data,label="Har")
        a=np.arange(Psi_dim)
        Al=a.tolist()
        qpe2.append(CHar2, Al)

        #CHar3 = add_control(CHar2,1, ctrl_state=1,label="CHar")
        vec2[l]=vec[l].evolve(qpe2)


    #Challenge defintion (first object)
    ch=vec[0]
    ch_out=vec2[0]


    q = QuantumRegister(Psi_dim, 'q')
    c = ClassicalRegister(Psi_dim, 'c')
    qpe2 = QuantumCircuit(q,c)
    vec[0]=(1/np.sqrt(2))*(vec[0]+vec[1])
    CHar=random_unitary(2**Psi_dim, seed=42)
    CHar2=UnitaryGate(CHar.data,label="Har")
    a=np.arange(Psi_dim)
    Al=a.tolist()
    qpe2.append(CHar2, Al)
    vec2[0]=vec[0].evolve(qpe2)


    #for j in range(2**Psi_dim,n_shots):

    #    q = QuantumRegister(Psi_dim, 'q')
    #
    #    c = ClassicalRegister(Psi_dim, 'c')
    #    qpe2 = QuantumCircuit(q,c)

        #on=np.random.uniform(0, pi)
        #on2=np.random.uniform(0, pi)

        #qpe2.ry(on,0)
        #qpe2.ry(on2,1)

    #    vec[j]=random_statevector(2**Psi_dim,seed=j)
        #Create a unitary database |Psi_in>, |Psi_out>
        ## Definition of the Heisenber 3d model
    #    CHar=random_unitary(2**Psi_dim, seed=42)
    #    CHar2=UnitaryGate(CHar.data,label="Har")
    #    a=np.arange(Psi_dim)
    #    Al=a.tolist()
    #    qpe2.append(CHar2, Al)

        #CHar3 = add_control(CHar2,1, ctrl_state=1,label="CHar")
    #    vec2[j]=vec[j].evolve(qpe2)
        #print(vec3[j])
        #print("-------------")
    
    return vec,vec2,ch,ch_out



import qutip as qp


if __name__ == "__main__":

# Create and set up circuit
######################################################## Initialization ####################################################################
    ancilla_N=2  #This is your depth K implicitly
    Psi_dim=1
    n_shots=100

    #A=np.arange(n_shots)
    vec=[0]*n_shots
    vec2=[0]*n_shots

    #or call Dataset1_qiskit for qiskit implementation via circuit of the dataset (instead of classical evolution ) ----------------- or Dataset_statevector
    vec,vec2,ch,ch_out=Dataset_statevector(ancilla_N,Psi_dim,n_shots) #Dataset in and out

    ####################################################################
    print(vec)
    print(ch)

    for trial in range(ancilla_N):

        q = QuantumRegister(ancilla_N+Psi_dim+1+Psi_dim, 'q')
        c = ClassicalRegister(Psi_dim, 'c')
        qpe2 = QuantumCircuit(q,c)

        #Initialize Psi and ancilla
        #Psi--------------
        Psi=ch
        aux=np.arange(ancilla_N,ancilla_N+Psi_dim)
        aux=aux.tolist()
        qpe2.initialize(Psi, aux)

        #Ancilla with |-> state
        for k in range(ancilla_N):
            qpe2.initialize([1/np.sqrt(2), -1/np.sqrt(2)], k)

        #Ancilla for measurement initialization
        qpe2.initialize([1/np.sqrt(2), -1/np.sqrt(2)], ancilla_N+Psi_dim)


        ################################ Stage 0 #####################
        #Define reference state input Phi_in_r
        Phi_in_r=vec[trial]

        ################################ Stage 1 ##################### define ancilla_N times the W_Block operation

        for j in range(ancilla_N):
            W_block(Phi_in_r,vec[j],Psi_dim,j,qpe2,ancilla_N) #j is the ancilla qubit number

        ############################### Stage 2 ###################### perform verification using an extra qubit
        anc_aux=ancilla_N+Psi_dim

        #You can use: Verification_block_batch or Verification_block

        qpe2,check=Verification_block_batch(anc_aux,Phi_in_r,Psi_dim,qpe2,ancilla_N)
    
        if (check==0):
            print("Trial that finished: "+str(trial))
            print("")
            break
        elif(trial==(n_shots-1)):
            print("Error the sample data is too short and does not represent Si")
            quit()

    ############## Code passed inside:
    #Perform swap test with respective Phi_reference_out, initialize in aux

    
    Phi_out_r=vec2[trial]
    aux2=np.arange(ancilla_N+Psi_dim+1,ancilla_N+Psi_dim+Psi_dim+1)
    aux2=aux2.tolist()
    qpe2.initialize(Phi_out_r, aux2)


    #Psi and Phi_out_r
    qpe2.swap(aux,aux2)
    #############################################
    # Create the inverse circuit 
    for j in range(ancilla_N-1, -1, -1) : #Reverse order
        print(j)
        print(vec2[j],Phi_out_r)
        W_block(vec2[j],Phi_out_r,Psi_dim,j,qpe2,ancilla_N) #j is the ancilla qubit number
    

    #First component 
    a=np.arange(Psi_dim)
    At=a.tolist()

    b=np.arange(ancilla_N,ancilla_N+Psi_dim, dtype=int) #Generalize vectors [counting_qubit]+[ancilla, ancilla+target]
    Bt=b.tolist() #Turn into a list to be an input for append

    qpe2.measure(Bt,At)

    sim_toro = Aer.get_backend('aer_simulator')
    shots = 1000
    t_qpe2 = transpile(qpe2, sim_toro,seed_transpiler=42)
    qobj = assemble(t_qpe2, shots=shots)
    results = sim_toro.run(qobj).result()
    finalm = results.get_counts()


    #print(qpe2)


    ############################### create a simulation of statistics for the orginal Psi ###########################
    

    print(qpe2)

    q = QuantumRegister(Psi_dim, 'q')
    c = ClassicalRegister(Psi_dim, 'c')
    qpe2 = QuantumCircuit(q,c)

    PsiOut=[0]*shots
    Psi_out=ch_out

    a=np.arange(Psi_dim)
    At=a.tolist()

    qpe2.initialize(Psi_out, At)

    qpe2.measure(At,At)

    sim_toro = Aer.get_backend('aer_simulator')
    shots = 1000
    t_qpe2 = transpile(qpe2, sim_toro,seed_transpiler=42)
    qobj = assemble(t_qpe2, shots=shots)
    results = sim_toro.run(qobj).result()
    finalm2 = results.get_counts()

    print("Compare both:")
    print(finalm)
    print("")
    print(finalm2)
    #qpe2.save_statevector()

    #backend = QasmSimulator()
    #backend_options = {'method': 'statevector'}
    #job = execute(qpe2, backend, backend_options=backend_options)
    #job_result = job.result()
    #prediction=job_result.get_statevector(qpe2)
    #prediction=np.array(prediction)

    #Short_pred=[0]*(2**Psi_dim)

    #Trim down the prediction to the state vector without ancilla
    
    ############################### Compare prediction with reality #############

    #print("")