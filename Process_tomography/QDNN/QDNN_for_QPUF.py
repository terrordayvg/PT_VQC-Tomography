import random

#initialization
import matplotlib.pyplot as plt
import numpy as np
import math

# importing Qiskit
from qiskit import transpile, assemble
from qiskit_aer import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import UnitaryGate

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


# Importing standard Qiskit modules
from qiskit import *
from qiskit.circuit import Parameter

# import basic plot tools
from qiskit.visualization import plot_histogram

# Import state tomography modules
#####from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import Statevector

# Import other modules
import numpy as np
import matplotlib.pyplot as plt
import cmath

import pennylane as qml


import scipy as sc
import qutip as qt
# further packages
from time import time
from random import sample
import matplotlib.pyplot as plt
import numpy as np

from qiskit.quantum_info import state_fidelity
import qiskit.quantum_info as qi
from qiskit.result import marginal_counts
import math



from multiprocessing.pool import ThreadPool

#############################################################################################################
####################################### All functions for the QDNN definition ###############################

#eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
def fidelityLine(qnnArch,Predrho, C1,C2,Testrho,total_t,n):
    kind = "randomUnitary"

    Testrhod=randomTestingData(total_t,list(Testrho))
    from qutip import fidelity

    
    dims1 = [2**n]
    dims2 = [2**n]
    dims = [dims1, dims2]

    for k in range(total_t):
        Predrho[k].dims = dims


    print("pred | test ===============")
    print(Predrho[0])
    print(Testrhod[0])
    print("")
    print("----------------------------------------------")

    
    fidMatrix=[[0 for k in range(total_t)] for r in range(total_t)]
    for k in range(total_t):
        for l in range(total_t):
            #print(Predrho[k],Testrhod[l])
            fidMatrix[k][l] = fidelity(Predrho[k],Testrhod[l])



    print("Fidelity Matrix")
    print("")
    print(fidMatrix)
    print("")
    print("---------------------------------------------------================================")
    

    # make fidelity plot
    plt.matshow(fidMatrix)
    plt.colorbar()
    plt.savefig(kind + '_' + str(total_t) + 'pairs_' + str(qnnArch[-1]) + 'Prediction_Real_Confusion_Fidelity_Matrix' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()







def fidelityMatrixRandomUnitary(qnnArch, numTrainingPairs, SIn, SOut):
    kind = "randomUnitary"
    # make a unitary for the training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    # create the training data
    ##trainingData = randomTrainingData(networkUnitary, numTrainingPairs)
    # create a sublist of the training data with the (un)supervised pairs (trainingDataSv and trainingDataUsv)
    # and its labels (listSv and listUsv)
    trainingData = randomTrainingData(networkUnitary, numTrainingPairs,SIn,SOut) #qt.Qobj(TData)


    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p

    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities_Heisenberg_3a4t_RandomInit_tut' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    #fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    #with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
    #          "wb") as fp:  # Pickling
    #    pickle.dump(fidelityMatrixAndTrainingData, fp)




def partialTraceKeep(obj, keep): # generalisation of ptrace(), partial trace via "to-keep" list
    # return partial trace:
    res = obj;
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

def partialTraceRem(obj, rem): # partial trace via "to-remove" list
    # prepare keep list
    rem.sort(reverse=True)
    keep = list(range(len(obj.dims[0])))
    for x in rem:
        keep.pop(x)
    res = obj;
    # return partial trace:
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

def swappedOp(obj, i, j):
    if i==j: return obj
    numberOfQubits = len(obj.dims[0])
    permute = list(range(numberOfQubits))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)

def tensoredId(N):
    #Make Identity matrix
    res = qt.qeye(2**N)
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def tensoredQubit0(N):
    #Make Qubit matrix
    res = qt.fock(2**N).proj() #for some reason ran faster than fock_dm(2**N) in tests
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res


def unitariesCopy(unitaries): # deep copyof a list of unitaries
    newUnitaries = []
    for layer in unitaries:
        newLayer = []
        for unitary in layer:
            newLayer.append(unitary.copy())
        newUnitaries.append(newLayer)
    return newUnitaries

def randomQubitUnitary(numQubits): # alternatively, use functions rand_unitary and rand_unitary_haar
    dim = 2**numQubits
    #Make unitary matrix
    res = np.random.normal(size=(dim,dim)) + 1j * np.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    #Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res

def measurements(qnnArch, unitaries, trainingData, R):
    feed = feedforward(qnnArch, unitaries, trainingData)
    prob = []
    for i in range(len(trainingData)):
        state = feed[i][-1]
        a = [] # probabilities with tr(E_i*rho)
        #print("")
        #print(len(qubit0mat))
        #print("")
        #print(qubit0mat)
        a.append(np.trace(np.array(qubit0mat*state)).real)
        a.append(np.trace(np.array(qubit1mat*state)).real)
        results = np.random.choice([1,-1], R, p=a)
        #number of measurements with outcome +1:
        k = 0
        for i in range(R):
            if results[i] == 1:
                k = k+1
        #empirical probability distribution:
        plus = k/R
        minus = (R - k)/R
        prob.append([plus, minus])
    return prob

def howManyWrong(qnnArch, unitaries, trainingData, R):
    prob = measurements(qnnArch, unitaries, trainingData, R)
    m = len(trainingData)
    x = 0
    for i in range(m):
        if trainingData[i][1] == qubit0:
            if prob[i][0] < 0.5:
                x = x + 1
        else:
            if prob[i][1] < 0.5:
                x = x + 1
    x = x/m
    return x

def randomQubitState(numQubits,SIn,SOut): # alternatively, use functions rand_ket and rand_ket_haar
    dim = 2**numQubits
    res = qt.Qobj(SIn)
    resO= qt.Qobj(SOut)

    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [1 for i in range(numQubits)]
    dims = [dims1, dims2]
    print(SIn,SOut,dims)

    res.dims = dims

    resO.dims= dims
    #Return
    return res,resO


def randomTestingData(N,Test): # generating a testing rho based on Qobj identation
    

    networkUnitary = randomQubitUnitary(qnnArch[-1])
    numQubits = len(networkUnitary.dims[0])
    dim = 2**numQubits
    #print(Test)
    #print("-------")

    Tclose=[[]]

    Penny=[[[0 for k in range(dim)] for r in range(dim)]for y in range(total_s)]

    trainingData=[]
    #Create training data pairs
    for i in range(N):
        for j in range(dim):
            for k in range(dim):
                Penny[i][j][k]=Test[i][j][0][k]

        print("")
        
        #print(Test[0][0][i])
        Testd=qt.Qobj(Penny[i])
        dims1 = [dim]
        dims2 = [dim]
        dims = [dims1, dims2]
        Testd.dims = dims
        #t,ut = randomQubitState(numQubits,SIn[i],SOut[i])
        #ut = unitary*t
        trainingData.append(Testd)
    #Return
    #print(trainingData)
    return trainingData


def randomTrainingData(unitary, N,SIn,SOut): # generating training data based on a unitary
    numQubits = len(unitary.dims[0])
    trainingData=[]
    #Create training data pairs
    for i in range(N):
        #print("----------------------------------------------------")
        #print("Generation N "+str(i)+":")
        t,ut = randomQubitState(numQubits,SIn[i],SOut[i])
        #ut = unitary*t
        trainingData.append([t,ut])
    #Return
    return trainingData


def randomNetwork(qnnArch, numTrainingPairs,SIn,SOut):
    assert qnnArch[0]==qnnArch[-1], "Not a valid QNN-Architecture."
    
    #Create the targeted network unitary and corresponding training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = randomTrainingData(networkUnitary, numTrainingPairs,SIn,SOut) #qt.Qobj(TData)
    #print(networkTrainingData)
    
    #Create the initial random perceptron unitaries for the network
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l-1]
        numOutputQubits = qnnArch[l]
        
        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = randomQubitUnitary(numInputQubits+1)
            if numOutputQubits-1 != 0: 
                unitary = qt.tensor(randomQubitUnitary(numInputQubits+1), tensoredId(numOutputQubits-1))
                unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)
    
    #Return
    return (qnnArch, networkUnitaries, networkTrainingData, networkUnitary)


#Creates generates the cost function based on <psi|rho|psi>
def costFunction(trainingData, outputStates):
    costSum = 0
    for i in range(len(trainingData)):
        costSum += trainingData[i][1].dag() * outputStates[i] * trainingData[i][1]
    return costSum.real/len(trainingData)


def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]

    #Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    #Multiply and tensor out input state
    return partialTraceRem(layerUni * state * layerUni.dag(), list(range(numInputQubits)))


def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    
    #Prepare needed states
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)

    #Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni
    
    #Multiply and tensor out output state
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)) )


def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0] * trainingData[x][0].dag()
        layerwiseList = [currentState]

        #print("")
        #print(unitaries)
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l-1]
    
    #Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        #Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)
        
        #Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)
        
        #Add to sum
        summ = summ + mat

    #Calculate the update matrix from the sum
    summ = (-ep * (2**numInputQubits)/(lda*len(trainingData))) * summ
    return summ.expm()


def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    
    #Tensor input state
    state = qt.tensor(storedStates[x][l-1], tensoredQubit0(numOutputQubits))
    
    #Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j+1):
        productUni = unitaries[l][i] * productUni
    
    #Multiply
    return productUni * state * productUni.dag()


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    
    #Calculate sigma state
    state = trainingData[x][1] * trainingData[x][1].dag()
    for i in range(len(qnnArch)-1,l,-1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    #Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)
    
    #Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j+1, numOutputQubits):
        productUni = unitaries[l][i] * productUni
        
    #Multiply
    return productUni.dag() * state * productUni


def makeUpdateMatrixTensored(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j):
    numInputQubits = qnnArch[l-1]
    numOutputQubits = qnnArch[l]
    
    res = makeUpdateMatrix(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j)
    if numOutputQubits-1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits-1))
    return swappedOp(res, numInputQubits, numInputQubits + j)


def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    
    #Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    #Cost calculation for given unitaries
    outputStates = [storedStates[k][-1] for k in range(len(storedStates))]
    plotlist = [[s], [costFunction(trainingData, outputStates)]]
    
    #Optional
    runtime = time()
    
    #Training of the Quantum Neural Network
    for k in range(trainingRounds):
        runtimet = time()

        #if alert>0 and k%alert==0: print("In training round "+str(k))
        print("In training round "+str(k))
        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)
        
        #Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l-1]
            numOutputQubits = qnnArch[l]
            
            #Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (makeUpdateMatrixTensored(qnnArch,currentUnitaries,trainingData,storedStates,lda,ep,l,j)* currentUnitaries[l][j])
        
        ### FEEDFORWARD
        #Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)
        
        #Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])
        plotlist[0].append(s*10)
        plotlist[1].append(costFunction(trainingData, outputStates))
        runtimet2 = time() - runtimet
        #print("Output_States")
        #print(outputStates)
        print("Finished training round --------"+ str(round(runtimet2, 2))+" seconds" )
        print("")

    
    #Optional
    print("")
   # print("CurrentUnitaries:")
   # print(currentUnitaries)
    print("")
    runtime = time() - runtime
    print("Trained "+str(trainingRounds)+" rounds for a "+str(qnnArch)+" network and "+str(len(trainingData))+" training pairs in "+str(round(runtime, 2))+" seconds")
    
    #Return
    return [plotlist, currentUnitaries]


def boundRand(D, N, n):
    return (n/N) + (N-n)/(N*D*(D+1)) * (D + min(n**2+1, D**2))


def subsetTrainingAvg(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, iterations, n, alertIt=0):
    costpoints = []
    
    for i in range(iterations):
        #if alertIt>0 and i%alertIt==0: print("n="+str(n)+", i="+str(i))
        #Prepare subset for training
        trainingSubset = sample(trainingData, n)
        
        #Train with the subset
        learnedUnitaries = qnnTraining(qnnArch, initialUnitaries, trainingSubset, lda, ep, trainingRounds)[1]
        storedStates = feedforward(qnnArch, learnedUnitaries, trainingData)
        outputStates = []
        for k in range(len(storedStates)):
            outputStates.append(storedStates[k][-1])
        
        #Calculate cost with all training data
        costpoints.append(costFunction(trainingData, outputStates))
    
    return sum(costpoints)/len(costpoints)


def noisyDataTraining(qnnArch, initialUnitaries, trainingData, noisyData, lda, ep, trainingRounds, numData, stepSize, alertP=0):
    noisyDataPlot = [[], []]
    
    i = 0
    while i <= numData:
        #if alertP>0: print("Currently at "+str(i/numData)+"% noisy data.")
        
        #Prepare mixed data for traing
        testData1 = sample(trainingData, numData - i)
        testData2 = sample(noisyData, i)
        if i==0: testData = testData1
        elif i==numData: testData = testData2
        else: testData = testData1 + testData2
        
        #Train with the mixed data
        learnedUnitaries = qnnTraining(qnnArch, initialUnitaries, testData, lda, ep, trainingRounds)[1]
        storedStates = feedforward(qnnArch, learnedUnitaries, trainingData)
        outputStates = []
        for k in range(len(storedStates)):
            outputStates.append(storedStates[k][-1])
        
        #Calculate cost with the real training data
        noisyDataPlot[0].append(i)
        noisyDataPlot[1].append(costFunction(trainingData, outputStates))
        
        i += stepSize
    
    return noisyDataPlot


#############################################################################################################
def not_a_haar_random_unitary():
    # Sample all parameters from their flat uniform distribution
    phi, theta, omega = 2 * np.pi * np.random.uniform(size=3)
    qml.Rot(phi, theta, omega, wires=0)
    return qml.state()

num_samples = 2021



# Used the mixed state simulator so we could have the density matrix for this part!
def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector."""
    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real
    return [ax, ay, az]

from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

# Samples of theta should be drawn from between 0 and pi

def haar_random_unitary():
    phi, omega = 2 * np.pi * np.random.uniform(size=2) # Sample phi and omega as normal
    theta = sin_sampler.rvs(size=1) # Sample theta from our new distribution
    qml.Rot(phi, theta, omega, wires=0)
    return qml.state()





from numpy.linalg import qr

def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)


def qr_haar_random_unitary():
    
    qml.QubitUnitary(qr_haar(2), wires=0)
    return qml.state()




def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit


def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates


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

    return int(t4),int(t5),int(t6)

from qiskit.circuit.add_control import add_control
from sympy import *


def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), m, j)
        qc.h(j)


def zz_cc_3D(delta, qubits,J):
    ''' Returns a circuit implementing ZZ(delta) for the 3D Heisenberg model applied to qubits '''
    zz = QuantumCircuit(4, name='ZZ')
    zz.cnot(qubits[0],qubits[1])
    zz.rz(2*delta*J, qubits[1])
    zz.cnot(qubits[0],qubits[1])
    return zz

def yy_cc_3D(delta, qubits,J):
    ''' Returns a circuit implementing YY(delta) for the 3D Heisenberg model applied to qubits '''
    yy = QuantumCircuit(4, name='YY')
    yy.rx(np.pi/2, [qubits[0],qubits[1]])
    ### ZZ ###
    yy.cnot(qubits[0],qubits[1])
    yy.rz(2*delta*J, qubits[1])
    yy.cnot(qubits[0],qubits[1])
    ##########
    yy.rx(-np.pi/2, [qubits[0],qubits[1]])
    return yy

def xx_cc_3D(delta, qubits,J):
    ''' Returns a circuit implementing XX(delta) for the 3D Heisenberg model applied to qubits '''
    xx = QuantumCircuit(4, name='XX')
    xx.ry(np.pi/2, [qubits[0],qubits[1]])
    ### ZZ ###
    xx.cnot(qubits[0],qubits[1])
    xx.rz(2*delta*J, qubits[1])
    xx.cnot(qubits[0],qubits[1])   
    ##########
    xx.ry(-np.pi/2, [qubits[0],qubits[1]])
    return xx


def Trotter_cc_3D(delta,J_v):
    ''' Create a Trotter step by appending XX, YY and ZZ circuits '''
    Trotter_cc = QuantumCircuit(4, name='Trotter')
    #print(J_v)
    #print("")
    
    Trotter_cc.append(zz_cc_3D(delta, [0,1],J_v[0]), [0,1,2,3])
    Trotter_cc.append(yy_cc_3D(delta, [0,1],J_v[1]), [0,1,2,3])
    Trotter_cc.append(xx_cc_3D(delta, [0,1],J_v[2]), [0,1,2,3])
    
    Trotter_cc.append(zz_cc_3D(delta, [1,2],J_v[3]), [0,1,2,3])
    Trotter_cc.append(yy_cc_3D(delta, [1,2],J_v[4]), [0,1,2,3])
    Trotter_cc.append(xx_cc_3D(delta, [1,2],J_v[5]), [0,1,2,3])   

    Trotter_cc.append(zz_cc_3D(delta, [2,3],J_v[6]), [0,1,2,3])
    Trotter_cc.append(yy_cc_3D(delta, [2,3],J_v[7]), [0,1,2,3])
    Trotter_cc.append(xx_cc_3D(delta, [2,3],J_v[8]), [0,1,2,3])   
    
    return Trotter_cc

def initialise(circuit, state, random=False):
    ''' Initialises the circuit based on the string state '''
    i = 0
    for num in state:
        if (random==True):
            on=np.random.uniform(0, np.pi)
            circuit.ry(on,i)
            
        elif (num == '1'):
            circuit.x(i)
        
        elif num == '+':
            circuit.h(i)
        elif num == '-':
            circuit.x(i)
            circuit.h(i)
        i+=1



def State_distribution2D(delta, shots, initial_state_qiskit, noise,random):
    ''' Measures the probability of being in a certain state of the computational basis '''
    dim=4
    Final_vectorx=[]
    Final_vectorz=[]
    Delta=[]

    delta=time 
    random=False
    noise=False
    x=np.arange(dim)
    results = np.zeros(len(delta))
    initial_state_qiskit="+"
    
    for t in range(len(delta)):
        qc_2D_spins = Heisenberg_2D(delta[t], initial_state_qiskit,random) 
        
        for j in range(n_qubits):
            qc_2D_spins.measure(j,j)
        
        if t==0:
            print(qc_2D_spins)
        
        # Run on QASM simulation
        if noise == True:
            t_qpe2 = transpile(qc_2D_spins,backend=simulator,coupling_map=coupling_map,basis_gates=basis_gates,initial_layout=[0,1])#,initial_layout=[15,12]            
            job_results = simulator.run(t_qpe2,shots=shots,noise_model=noise_model).result()
            

        else:
            t_qpe2 = transpile(qc_2D_spins,backend=simulator)                      
            job_results = simulator.run(t_qpe2,shots=shots).result()
        
        # Get counts
        counts_spins = job_results.get_counts()
        prob=[0]*dim

   
        for l in range(4):
            de=decimalToBinary(l)
            ext=str(de[0])+str(de[1])
            try:
                prob[l]=counts_spins[ext]

        
            except:
                prob[l]=0
    
    
            Final_vectorx.append(x[l])            #X
            Final_vectorz.append(prob[l]/shots)   #Z
            Delta.append(delta[t])                #Y  for the 3D plot

       
                
    return Final_vectorx,Delta,Final_vectorz

def Backbone(qis_seed,n_ancilla=2,n_target=2, seedH=2):
    
    """
    Generalized Backbone function with arbitrary number of ancilla and target qubits.
    
    Args:
        qis_seed (int): Seed for reproducibility
        n_ancilla (int): Number of ancilla qubits
        n_target (int): Number of target qubits
        
    Returns:
        Vec_in (np.array): Initial statevector
        Vec_out (np.array): Statevector after applying the circuit
    """
    np.random.seed(qis_seed)
    
    total_qubits = n_ancilla + n_target
    q = QuantumRegister(total_qubits, 'q')
    c = ClassicalRegister(n_ancilla, 'c')
    qc = QuantumCircuit(q, c)
    
    # Random rotation for target qubits
    for tq in range(n_ancilla, total_qubits):
        qc.ry(np.random.uniform(0, pi), tq)
    
    # Random unitary on target qubits
    Init = random_unitary(2**n_target, seed=qis_seed)
    Init_gate = UnitaryGate(Init.data, label="Init")
    target_qubits = list(range(total_qubits - n_target, total_qubits))
    qc.append(Init_gate, target_qubits)
    
    # Controlled random unitary for QPE-like evolution
    CHar = random_unitary(2**n_target, seed=seedH)
    CHar_gate = UnitaryGate(CHar.data, label="CHar")
    CHar_controlled = add_control(CHar_gate, 1, ctrl_state=1, label="C-CHar")
    
    # Apply Hadamards to ancilla qubits
    for aq in range(n_ancilla):
        qc.h(aq)
    
    # Apply controlled operations in QPE style
    repetitions = 1
    for aq in range(n_ancilla):
        for _ in range(repetitions):
            qc.append(CHar_controlled, [aq] + list(range(n_ancilla, total_qubits)))
        repetitions *= 2
    
    # Inverse QFT on ancilla
    qft_dagger(qc, n_ancilla)
    
    # Get initial statevector
    stateIn = Statevector(qc)
    
    # Simulate
    backend = Aer.get_backend("statevector_simulator")
    qc_t = transpile(qc, backend)
    result = backend.run(qc_t, seed_transpiler=qis_seed).result()
    
    stateOut = result.get_statevector()

    
    return np.array(stateIn), np.array(stateOut)


def Backbone2(qis_seed,n_ancilla=2,n_target=2,seedH=2):
    
    """
    Generalized Backbone function with arbitrary number of ancilla and target qubits.
    
    Args:
        qis_seed (int): Seed for reproducibility
        n_ancilla (int): Number of ancilla qubits
        n_target (int): Number of target qubits
        
    Returns:
        Vec_in (np.array): Initial statevector
        Vec_out (np.array): Statevector after applying the circuit
    """
    np.random.seed(qis_seed)
    
    total_qubits = n_ancilla + n_target
    q = QuantumRegister(total_qubits, 'q')
    c = ClassicalRegister(n_ancilla, 'c')
    qc = QuantumCircuit(q, c)
    
    # Random rotation for target qubits
    for tq in range(n_ancilla, total_qubits):
        qc.ry(np.random.uniform(0, pi), tq)
    
    # Random unitary on target qubits
    Init = random_unitary(2**n_target, seed=qis_seed)
    Init_gate = UnitaryGate(Init.data, label="Init")
    target_qubits = list(range(total_qubits - n_target, total_qubits))
    qc.append(Init_gate, target_qubits)
    
    # Controlled random unitary for QPE-like evolution
    CHar = random_unitary(2**n_target, seed=seedH)
    CHar_gate = UnitaryGate(CHar.data, label="CHar")
    CHar_controlled = add_control(CHar_gate, 1, ctrl_state=1, label="C-CHar")
    
    # Apply Hadamards to ancilla qubits
    for aq in range(n_ancilla):
        qc.h(aq)
    
    # Apply controlled operations in QPE style
    repetitions = 1
    for aq in range(n_ancilla):
        for _ in range(repetitions):
            qc.append(CHar_controlled, [aq] + list(range(n_ancilla, total_qubits)))
        repetitions *= 2
    
    # Inverse QFT on ancilla
    qft_dagger(qc, n_ancilla)
    
    # Get initial statevector
    stateIn = Statevector(qc)
    
    # Simulate
    backend = Aer.get_backend("statevector_simulator")
    qc_t = transpile(qc, backend)
    result = backend.run(qc_t, seed_transpiler=qis_seed).result()
    
    stateOut = result.get_statevector()

    rho_F = np.array(qi.DensityMatrix(stateOut))

    return np.array(stateIn), np.array(stateOut),rho_F




# wrapper function for task
def task_wrapper(args):
    # call task() and unpack the arguments
    return Backbone(*args)

def task_wrapper2(args):
    # call task() and unpack the arguments
    return Backbone2(*args)


def plot_cost_function(x, y, xlabel="Iterations", ylabel="Cost", title="Cost Function vs. iterations",
                       line_color='royalblue', save_path='Costfunct.png',
                       show=True, save=True):
    """
    Create and optionally save a styled cost function plot.

    Parameters
    ----------
    x : array-like
        X-axis data (e.g., s values)
    y : array-like
        Y-axis data (e.g., cost values)
    xlabel : str, optional
        Label for the X-axis
    ylabel : str, optional
        Label for the Y-axis
    title : str, optional
        Title of the plot
    line_color : str, optional
        Color of the plot line
    save_path : str, optional
        File path to save the plot
    show : bool, optional
        Whether to display the plot
    save : bool, optional
        Whether to save the plot as a file
    """

    plt.close('all') #close all open window instances

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color=line_color, linewidth=2.5, marker='o', markersize=5, label='Cost Function')

    plt.xlabel(xlabel, fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)

    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()



if __name__ == "__main__":

    # Parameters to define
    seedc=100
    ntrial=0
    m=0
    cores=1

    qnnArch=[2,2]   #Arquitecture of the NN

    if qnnArch[0]!=qnnArch[1]:
        print("Error: Dim QNN should be the same for both layers eg. 2,2 or 4,4")
        exit()

    total_s=2    #Number of Shots training
    total_t=2    #Number of Shots testing
    seedH=442     #Seed of Harr random unitary generation

    workers=1 #Numbers of workers for paralellization
    ############################################################################
    #Qutip initial definiton 0 and 1 states in tensor and normal forms
    #ket states
    qubit0 = qt.basis(2,0)
    qubit1 = qt.basis(2,1)
    #density matrices
    qubit0mat = qubit0 * qubit0.dag()
    qubit1mat = qubit1 * qubit1.dag()

    print("-------------------------------------------------------------------------------")
    print("Simulation of "+str(qnnArch[0])+" qubits")
    print("")

    print("Data Generation step:")
    print("--------------------------")

    ###############################################################################
    off_set=100000
    qis_seed=np.arange(total_s)
    qis_seed2=np.arange(off_set,off_set+total_t)

  
    args = [[qis_seed[i],int(qnnArch[0]/2),int(qnnArch[0]/2), seedH ] for i in range(total_s)]
    args2 =[[qis_seed2[i],int(qnnArch[0]/2),int(qnnArch[0]/2), seedH]  for i in range(total_t)]


    print("Generation of training data:")
    with Pool(cores) as pool:
    # Pool map for the seed in paralel.
        Training_Data=pool.map(task_wrapper, args)
        pool.close()
        pool.join()

    print("Generation of testing data:")
    with Pool(cores) as pool:
    # Pool map for the seed in paralel.
        Testing_Data=pool.map(task_wrapper2, args2)
        pool.close()
        pool.join()


    print("Training Data Generation:")
    Pos_c1=[[[0] for r in range(len(Training_Data[0][0]))]for y in range(total_s)]
    Pos_c2=[[[0] for r in range(len(Training_Data[0][0]))]for y in range(total_s)]

    Pos_ct1=[[[0] for r in range(len(Training_Data[0][0]))]for y in range(total_t)]
    Pos_ct2=[[[0] for r in range(len(Training_Data[0][0]))]for y in range(total_t)]
    Pos_ct3=[[[0] for r in range(len(Training_Data[0][0]))]for y in range(total_t)]


    print("")

    datasets = [
    (Pos_c1, Training_Data, 0, total_s),
    (Pos_c2, Training_Data, 1, total_s),
    (Pos_ct1, Testing_Data, 0, total_t),
    (Pos_ct2, Testing_Data, 1, total_t),
    (Pos_ct3, Testing_Data, 2, total_t)
    ]

    for pos, data, idx, total in datasets:
        for i in range(total):
            for k in range(len(data[0][idx])):
                pos[i][k][0] = data[i][idx][k]  

    print("")
    print("Definition of QDNN and Training step:")
    print("--------------------------")
    print("")
    print("N of training pairs:"+str(total_s)+" --- Number of Testing pairs: " +str(total_t))
    print("")
    print("--------------------------")
    print("Fidelity between all training set final state vectors")
    print("--------------------------")

    network33 = randomNetwork(qnnArch, total_s, Pos_c1, Pos_c2)

    #qnnArch|  initialUnitaries| trainingData|lda| ep  |trainingRounds| alert=0
    plotlist33, unitaries33 = qnnTraining(network33[0], network33[1], network33[2], 1, 0.1, 15,1)


    #######################################################################################################
    ######################################## Post training steps ##########################################
    #######################################################################################################

    network33t = randomNetwork(qnnArch, total_t, Pos_ct1, Pos_ct2)        # 1- Generate a random network with data from test set
    storedStates = feedforward(network33t[0], unitaries33, network33t[2]) # 2- Knowing initial unitaries from training set, generate predicted state vector

    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])

    fidelityLine(qnnArch,outputStates,Pos_ct1,Pos_ct2,Pos_ct3,total_t,qnnArch[0])    # 3- Compare the predicted to tested state vector using fidelity and fidelity matrix


    print("")
    print("Cost function analysis:")
    print("")
    print(plotlist33[1])
    print("----------------------------")
    plot_cost_function(plotlist33[0], plotlist33[1])
    


    