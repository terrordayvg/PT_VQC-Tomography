import numpy as np
import matplotlib.pyplot as plt


import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

# importing Qiskit
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions import *

# import basic plot tools
from qiskit.visualization import plot_histogram
# Let's see how it looks:
from numpy.random import uniform
from numpy.random import random
from numpy import pi
import numpy as np
from numpy.linalg import qr
import matplotlib.pyplot as plt
import math

from qiskit.quantum_info import random_unitary


from torch.nn import Linear
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.constraints import maxnorm
from keras.layers.core import Dense, Activation, Dropout

from multiprocessing import Pool
from time import time
from itertools import repeat

import pennylane as qml

from qiskit.circuit.add_control import add_control
from qiskit.providers.aer import QasmSimulator, AerSimulator

############################################################################################
############### Last modified: 12/07/22 author: Vladlen G. #################################
############################################################################################
############### Code objective is to study classical tomography of phase based QPUF ########
############################################################################################

############### Edit for later : This codes needs to be simplified down, cut libraries and optimize code ###########################

def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    np.random.seed(32242)
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    zul=np.dot(Q,Lambda)
    
    return zul


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
    elif(ip_val<4):
        t3,t4,t5,t6=b
        t4=0
        t3=0
        t2=0
        t1=0
        t0=0
    elif(ip_val<8):
        t2,t3,t4,t5,t6=b
        t3=0
        t2=0
        t1=0
        t0=0
    elif(ip_val<16):
        t1,t2,t3,t4,t5,t6=b
        t2=0
        t1=0
        t0=0

    else:
        t0,t1,t2,t3,t4,t5,t6=b
    return int(t4),int(t5),int(t6)


def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), m, j)
        qc.h(j)


def QuantumCircuit3(seeds):
    # Random input and random haar unitary, try to learn from this just from input and dataset classical output
    
    dim_ancilla=3
    dim_target=3

    print(seeds)
    count=False 
    tot=[0]*1
    #tot=[0]*len(seeds)
    X_l=[[0 for x in range(dim_ancilla)] for y in range(1)] 
    #X_l=[[0 for x in range(5)] for y in range(len(seeds))] 

    X_l_tot=[[0 for x in range(1)] for y in range(1)] 
    #X_l_tot=[[0 for x in range(1)] for y in range(len(seeds))] 
    
    for i in range(1):
        

        # --- Circuit definition ---
        
        q = QuantumRegister(6, 'q')
        c = ClassicalRegister(3, 'c')
        qpe2 = QuantumCircuit(q,c)
        
        

        np.random.seed(seeds)

        on=uniform(0, pi)
        on2=uniform(0, pi)
        on3=uniform(0, pi)
        on4=uniform(0, pi)
        on5=uniform(0, pi)

        for l in range(3):
            qpe2.h(l)
        

        qpe2.ry(on,3)
        qpe2.ry(on2,4)
        qpe2.ry(on3,5)




        CHar=random_unitary(8, seed=42)
        CHar2=UnitaryGate(CHar.data,label="Har")
        gate4x4 = add_control(CHar2,1, ctrl_state=1,label="CHar")

        
        
        
        qpe2.barrier()
       
        repetitions=1
        for counting_qubit in range(3):
            for j in range(repetitions):
                 qpe2.append(gate4x4, [counting_qubit,3,4,5])
        #        qpe2.cp(pi/3,5+counting_qubit,counting_qubit)
            repetitions *= 2

        qft_dagger(qpe2,3)

        
        qpe2.measure([0,1,2],[0,1,2])
      

        # ---------------------------



        aer_sim = Aer.get_backend('aer_simulator')
        shots = 4060
        t_qpe2 = transpile(qpe2, aer_sim, seed_transpiler=42)
        result2 = aer_sim.run(t_qpe2,shots=shots).result()
        result = result2.get_counts()
        exp_a=2**dim_ancilla
        prob=[0]*exp_a
        for z in range(exp_a):
            num=decimalToBinary(z)
            ext=str(num[0])+str(num[1])+str(num[2])
            
            try:
                prob[z]=result[ext]
        
            except:
                prob[z]=0



        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / shots
        
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        tot[i]=prob
        
        X_l[i][0]=on
        X_l[i][1]=on2
        X_l[i][2]=on3
           
    

    return tot,X_l




def define_model(dim_ancilla):#, batch_normalization):
    '''
    This function defines different models and returns the model and name of the subdirectory
    '''
    print("-------\nRunning functional API!\n")
    init="glorot_uniform"
    activation_function='relu'
    model = Sequential()


    model.add(Dense(128, activation='relu', kernel_initializer=init, input_shape=(2**dim_ancilla,)))
    
    #dense2 = Dense(250, activation=activation_function, kernel_initializer=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2))(dropout1)
    model.add(Dense(128, activation=activation_function, kernel_initializer='he_uniform'))


    model.add(Dense(dim_ancilla))
    #dense4 = Dense(250, activation=activation_function, kernel_initializer=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2))(dropout3)
    #dense4 = Dense(250, activation=activation_function, kernel_initializer=init)(dropout3)
    #dropout4 = Dropout(0.1)(dense4)

    #dense5 = Dense(250, activation=activation_function, kernel_initializer=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2))(dropout4)
    #dense5 = Dense(250, activation=activation_function, kernel_initializer=init)(dropout4)
    #dropout5 = Dropout(0.1)(dense5)
    
    #dense6 = Dense(250, activation=activation_function, kernel_initializer=init, W_regularizer=l1l2(l1=l1, l2=l2), activity_regularizer=activity_l1l2(l1=activity_l1, l2=activity_l2))(dropout5)
    #dense6 = Dense(250, activation=activation_function, kernel_initializer=init)(dropout5)
    #dropout6 = Dropout(0.1)(dense6)

    

    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer="adam", loss="mse")

    return model


from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as pyplot


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.yscale('log')

    # plot accuracy
    #pyplot.subplot(212)
    #pyplot.title('Classification Accuracy')
    #pyplot.plot(history.history['accuracy'], color='blue', label='train')
   # pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    

    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')

    pyplot.show()
    pyplot.close()

# wrapper function for task
def task_wrapper(args):
    # call task() and unpack the arguments
    return QuantumCircuit3(args)

def Test_reference(ci):


######################################################## CIRCUIT ####################################################################
# Create and set up circuit
######################################################## Initialization ####################################################################

    q = QuantumRegister(6, 'q')
    c = ClassicalRegister(3, 'c')
    qpe2 = QuantumCircuit(q,c)

    #print(ci)
    # Try to see how fidelity changes if you apply the random initialization method, does it change or no?
    #qpe2.ry(on6,1)
    qpe2.ry(ci[0],3)
    qpe2.ry(ci[1],4)
    qpe2.ry(ci[2],5)


    #This willl be an initialization for the Heisenber 3d model
    
    ######################################################################################################################
    ## Definition of the Heisenber 3d model
    CHar=random_unitary(8, seed=42)
    CHar2=UnitaryGate(CHar.data,label="Har")
    CHar3 = add_control(CHar2,1, ctrl_state=1,label="CHar")


        
    for qubit in range(3):
        qpe2.h(qubit)



    repetitions=1
    for counting_qubit in range(3):
        for j in range(repetitions):
            qpe2.append(CHar3, [counting_qubit,3,4,5])
 #          qpe2.cp(angle, counting_qubit, 5+counting_qubit);
        repetitions *= 2


            
#       for counting_qubit in range(5):
##          qpe2.append(gate4x4, [5+counting_qubit, counting_qubit])
#       qpe2.append(gate4x4, [5,6,7,8,9,0,1,2,3,4])
        
    qft_dagger(qpe2,3)
    #print("here6")


    qpe2.measure([0,1,2],[0,1,2]) ################################################## O que sai daqui devia de ser o mesmo que se fizessemos o measurement de no marginal - for some reason that's not the case.

    qpe2.reset([0,1,2])

    for qubit in range(3):
        qpe2.h(qubit)

    #           print("here6")
                
    #           print("here7")
    #           for counting_qubit in range(5):
    #               qpe2.append(gate4x4, [5+counting_qubit, counting_qubit])
    repetitions = 1
    for counting_qubit in range(3):
        for i in range(repetitions):
            qpe2.append(CHar3, [counting_qubit,3,4,5])
    #                   qpe2.append(gate4x4, [5+counting_qubit,counting_qubit])
#                       qpe2.cp(angle, counting_qubit, 5+counting_qubit);
        repetitions *= 2
                
    #           qpe2.append(gate4x4, [5,6,7,8,9,0,1,2,3,4])
    #           print("here8")
    qft_dagger(qpe2,3)


    qpe2.measure([0,1,2],[0,1,2])


    #####################################################################################################################################
    size=4060
    Pos_m1=[0]*size
    Pos_m2=[0]*size
    qis_seed=np.arange(size)
    #Paralellize task for every seed generation vs result.
    shots=1
    aer_sim = Aer.get_backend('aer_simulator')
    t_qpe2 = transpile(qpe2, aer_sim, seed_transpiler=42)
    for h in range(size):
        results = aer_sim.run(t_qpe2,seed_simulator=qis_seed[h],shots=shots).result()
        finalm = results.get_counts()
    


    #print(qis_seed,finalm,simp_counts1)

        for l in range(8):
            num=decimalToBinary(l)
            ext=str(num[0])+str(num[1])+str(num[2])
            try:
          
                if (finalm[ext] != 0):
                    Pos_m2[h]=l

            except:
                continue

           #pdf gen, ci, gen shots, verification shots
    return Pos_m2

def Reference(seedV,dim_ancilla):

    tot=[0]*1
    #tot=[0]*len(seeds)
    X_l=[0 for x in range(dim_ancilla)]  

############################ State position simulation #################################

    np.random.seed(seedV) # Seed for user is fixed (int not vector)
    #print(qis_seed)
    
    on=np.random.uniform(0, pi)
    on2=np.random.uniform(0, pi)
    on3=np.random.uniform(0, pi)
    on4=np.random.uniform(0, pi)
    on5=np.random.uniform(0, pi)
    on6=np.random.uniform(0, pi)
    #print(on,on2,on3,on4,on5)





######################################################## CIRCUIT ####################################################################
# Create and set up circuit
######################################################## Initialization ####################################################################

    q = QuantumRegister(6, 'q')
    c = ClassicalRegister(3, 'c')
    qpe2 = QuantumCircuit(q,c)


    # Try to see how fidelity changes if you apply the random initialization method, does it change or no?
    #qpe2.ry(on6,1)
    qpe2.ry(on,3)
    qpe2.ry(on2,4)
    qpe2.ry(on3,5)


    #This willl be an initialization for the Heisenber 3d model
    


    ######################################################################################################################
    ## Definition of the Heisenber 3d model
    CHar=random_unitary(8, seed=42)
    CHar2=UnitaryGate(CHar.data,label="Har")
    CHar3 = add_control(CHar2,1, ctrl_state=1,label="CHar")


        

        


    for qubit in range(3):
        qpe2.h(qubit)



    repetitions=1
    for counting_qubit in range(3):
        for j in range(repetitions):
            qpe2.append(CHar3, [counting_qubit,3,4,5])
 #          qpe2.cp(angle, counting_qubit, 5+counting_qubit);
        repetitions *= 2


            
#       for counting_qubit in range(5):
##          qpe2.append(gate4x4, [5+counting_qubit, counting_qubit])
#       qpe2.append(gate4x4, [5,6,7,8,9,0,1,2,3,4])
        
    qft_dagger(qpe2,3)
    #print("here6")


    qpe2.measure([0,1,2],[0,1,2]) ################################################## O que sai daqui devia de ser o mesmo que se fizessemos o measurement de no marginal - for some reason that's not the case.

######################################################## Verification 1 ####################################################################




    
    #sim_toro = Aer.get_backend('aer_simulator')
    #sim_toro = AerSimulator.from_backend(device_backend)

    shots = 4060
    qis_seed=np.arange(shots)

   # t_qpe2 = transpile(qpe2, sim_toro,seed_transpiler=42)

# ------------------------------------------------ Backend and simulator executable --------------------------------------------------- 
    aer_sim = Aer.get_backend('aer_simulator')
    t_qpe2 = transpile(qpe2, aer_sim, seed_transpiler=42)
    result2 = aer_sim.run(t_qpe2,shots=shots).result()
    result = result2.get_counts()
    exp_a=2**dim_ancilla
    prob=[0]*exp_a
    for z in range(exp_a):
        num=decimalToBinary(z)
        ext=str(num[0])+str(num[1])+str(num[2])
            
        try:
            prob[z]=result[ext]
        
        except:
            prob[z]=0



        
    counts = np.array(list(result.values()))
    states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
    probabilities = counts / shots
        
        # Get state expectation
    expectation = np.sum(states * probabilities)
        
    tot[0]=prob
        
    X_l[0]=on
    X_l[1]=on2
    X_l[2]=on3    

    ########################################################## Verification code #######################################################


######################################################## CIRCUIT ####################################################################
# Create and set up circuit
######################################################## Initialization ####################################################################

    q = QuantumRegister(6, 'q')
    c = ClassicalRegister(3, 'c')
    qpe2 = QuantumCircuit(q,c)


    # Try to see how fidelity changes if you apply the random initialization method, does it change or no?
    #qpe2.ry(on6,1)
    qpe2.ry(on,3)
    qpe2.ry(on2,4)
    qpe2.ry(on3,5)



    for qubit in range(3):
        qpe2.h(qubit)



    repetitions=1
    for counting_qubit in range(3):
        for j in range(repetitions):
            qpe2.append(CHar3, [counting_qubit,3,4,5])
 #          qpe2.cp(angle, counting_qubit, 5+counting_qubit);
        repetitions *= 2


            
#       for counting_qubit in range(5):
##          qpe2.append(gate4x4, [5+counting_qubit, counting_qubit])
#       qpe2.append(gate4x4, [5,6,7,8,9,0,1,2,3,4])
        
    qft_dagger(qpe2,3)
    #print("here6")


    qpe2.measure([0,1,2],[0,1,2]) ################################################## O que sai daqui devia de ser o mesmo que se fizessemos o measurement de no marginal - for some reason that's not the case.

    qpe2.reset([0,1,2])

    for qubit in range(3):
        qpe2.h(qubit)

    #           print("here6")
                
    #           print("here7")
    #           for counting_qubit in range(5):
    #               qpe2.append(gate4x4, [5+counting_qubit, counting_qubit])
    repetitions = 1
    for counting_qubit in range(3):
        for i in range(repetitions):
            qpe2.append(CHar3, [counting_qubit,3,4,5])
    #                   qpe2.append(gate4x4, [5+counting_qubit,counting_qubit])
#                       qpe2.cp(angle, counting_qubit, 5+counting_qubit);
        repetitions *= 2
                
    #           qpe2.append(gate4x4, [5,6,7,8,9,0,1,2,3,4])
    #           print("here8")
    qft_dagger(qpe2,3)


    qpe2.measure([0,1,2],[0,1,2])


    #####################################################################################################################################
    size=shots
    Pos_m2=[0]*size
    #Paralellize task for every seed generation vs result.
    shots=1
    for h in range(size):
        results = aer_sim.run(t_qpe2,seed_simulator=qis_seed[h],shots=shots).result()
        finalm = results.get_counts()
    


    #print(qis_seed,finalm,simp_counts1)

        for l in range(8):
            num=decimalToBinary(l)
            ext=str(num[0])+str(num[1])+str(num[2])
            try:
          
                if (finalm[ext] != 0):
                    Pos_m2[h]=l

            except:
                continue

           #pdf gen, ci, gen shots, verification shots
    return tot,X_l,Pos_m2



def Data_creation(dim_ancilla,pow_a,sizextr,sizex,workers):


    end1 = time()

    off_set=100
    #offs=151216124
    qis_seed=np.arange(sizex)

    args = [qis_seed[i] for i in range(sizex)]

    print("Test data generation:")
    
    print("-------------------------------------------")

    end1 = time()

    with Pool(workers) as pool:
    # Pool map for the seed in paralel.
        result=pool.map(task_wrapper, args)

        

        pool.close()
    # wait for all issued tasks to complete
        pool.join()


    X_t=[[0 for x in range(pow_a)] for y in range(sizex)] 
    Y_t=[[0 for x in range(dim_ancilla)] for y in range(sizex)] 
    

    for i in range(sizex):
        for j in range(pow_a):

            X_t[i][j]=result[i][0][0][j]


    for i in range(sizex):
        for j in range(dim_ancilla):
            Y_t[i][j]=result[i][1][0][j]

    print("X_t:")
    

    print("Y_t:")
    

    print("Size of result:")
    #print(len(result))

    qis_seed2=np.arange(off_set,off_set+sizextr)
    args = [qis_seed2[i] for i in range(sizextr)]


    print("Train data generation:")
    print("-------------------------------------------")

    with Pool(workers) as pool:
    # Pool map for the seed in paralel.
        result2=pool.map(task_wrapper, args)

        

        pool.close()
    # wait for all issued tasks to complete
        pool.join()


    print("Result: ")
    

    X_tr=[[0 for x in range(pow_a)] for y in range(sizextr)] 
    Y_tr=[[0 for x in range(dim_ancilla)] for y in range(sizextr)] 
    

    for i in range(sizextr):
        for j in range(pow_a):

            X_tr[i][j]=result2[i][0][0][j]


    for i in range(sizextr):
        for j in range(dim_ancilla):
            Y_tr[i][j]=result2[i][1][0][j]

    print("X_tr:")
    

    print("Y_tr:")
    

    print("Size of result:")
    print(len(result2))




    #X_tr,Y_tr=QuantumCircuit2(X_train)

    end2=time()

    timeend=end2-end1
    print("       ")
    print("Time of execution:")
    print(str(timeend)+"s")

    return X_t,Y_t,X_tr,Y_tr,


def plot_confusion_matrix(a,cm, classes, normalize=True, title='Normalized confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + str(a))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig("Confusion_Matrix"+str(a)+".pdf")
    plt.show()



# run the test harness for evaluating a model##########################################################################################################
def run_test_harness():

    #Parameters
    dim_ancilla=3
    pow_a=2**dim_ancilla
    sizextr=10 #training size
    sizex=10 #testing size
    workers=1 #paralellization

    print("Starting....")
    # load dataset
    testX,testY,trainX,trainY=Data_creation(dim_ancilla,pow_a,sizextr,sizex,workers)
    
    print("Passed data creation")

    

    model = define_model(dim_ancilla)
    # fit model
    history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_data=(testX, testY), verbose=0)
    # evaluate model
    #a, acc = model.evaluate(testX, testY, verbose=0)
    #print(a)
    print("----------------")
    predY = model.predict(testX)

    from sklearn.metrics import mean_squared_error

    print("DNN prediction step vs reality:-------------------------------------")
    print("Prediction Y:")
    print(predY)
    print("Reality ")
    print(testY)
    print(testY[:][0])
    print("c0 MSE:%.4f" % mean_squared_error(testY, predY))
    print("---------------------------------------------------------------------")

    #Create a prediction
    ####################################################################################
    seed=1251512431
    X_l,Y_l,Ref2=Reference(seed,dim_ancilla)
    print(X_l) #Probability distribution
    print("")
    print(Y_l) #Ci distribition
    print("")
    print(Ref2)
    print("")

    # Predict ci
    predY1 = model.predict(X_l)
    print("PREDY1")
    print(predY1[0])
    print("Pred to real MSE:%.4f" % mean_squared_error(Y_l,predY1[0]))

    #Run new simulation 
    Pos_c2l=Test_reference(predY1[0])

    #Compare the performance from the real and prediction just from the initialization

    print("DNN: Dim ancilla:"+str(dim_ancilla)+" target: 3 ancilla, method: seed tracking:")
    print("----")
    print("Reference:")
    print(" ")
    print(Ref2)
    print("")
    print("DNN Eve:")
    print(" ")
    print(Pos_c2l)
    print(" ")
    print("----")

    ####################################################################################



    c0_t=[0]*len(testY)
    c1_t=[0]*len(testY)
    c2_t=[0]*len(testY)
    c3_t=[0]*len(testY)
    c4_t=[0]*len(testY)


    c0_p=[0]*len(testY)
    c1_p=[0]*len(testY)
    c2_p=[0]*len(testY)
    c3_p=[0]*len(testY)
    c4_p=[0]*len(testY)



    csize=32
    # For confusion matrix class distribution 63 classes - 0.0 to 6.3 

    CYdec0=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    CYdec1=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    CYdec2=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    CYdec3=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    CYdec4=[[0 for x in range(csize+1)] for y in range(len(testY))] 

    # Prediction
    PCYdec0=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    PCYdec1=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    PCYdec2=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    PCYdec3=[[0 for x in range(csize+1)] for y in range(len(testY))] 
    PCYdec4=[[0 for x in range(csize+1)] for y in range(len(testY))] 

    # ROC curve
    tec0=[0]*len(testY)
    pec0=[0]*len(testY)

    for i in range(len(testY)):
        c0_t[i]=testY[i][0]-predY[i][0]
        c1_t[i]=testY[i][1]-predY[i][1]
        c2_t[i]=testY[i][2]-predY[i][2]
        #c3_t[i]=testY[i][3]-predY[i][3]
        #c4_t[i]=testY[i][4]-predY[i][4]


        c0_p[i]=predY[i][0]
        c1_p[i]=predY[i][1]
        c2_p[i]=predY[i][2]
        #c3_p[i]=predY[i][3]
        #c4_p[i]=predY[i][4]

        d0=int(round(testY[i][0],1)*10)
        d1=int(round(testY[i][1],1)*10)
        d2=int(round(testY[i][2],1)*10)
        #d3=int(round(testY[i][3],1)*10)
        #d4=int(round(testY[i][4],1)*10)
        #print("D0"+str( d0), i)
        

        Pd0=int(round(predY[i][0],1)*10)
        Pd1=int(round(predY[i][1],1)*10)
        Pd2=int(round(predY[i][2],1)*10)
        #Pd3=int(round(predY[i][3],1)*10)
        #Pd4=int(round(predY[i][4],1)*10)
        
        #print("PD0"+str( Pd0))
        

        tec0[i]=d0
        pec0[i]=Pd0

        if (d0==20):
            tec0[i]=1
        else:
            tec0[i]=0

        if (Pd0==20):
            pec0[i]=1
        else:
            pec0[i]=0

        # For class distribution discretization put a 1 in each 1 of 63 classes aprox to 1st decimal point
        CYdec0[i][d0]=1
        CYdec1[i][d1]=1
        CYdec2[i][d2]=1
        #CYdec3[i][d3]=1
        #CYdec4[i][d4]=1

               
        if (Pd0>csize):
            Pd0=csize
        if (Pd0<0):
            Pd0=0

        if (Pd1>csize):
            Pd1=csize
        if (Pd1<0):
            Pd1=0

        if (Pd2>csize):
            Pd2=csize
        if (Pd2<0):
            Pd2=0

        #if (Pd3>csize):
        #    Pd3=csize
        #if (Pd3<0):
        #    Pd3=0

        #if (Pd4>csize):
        #    Pd4=csize
        #if (Pd4<0):
        #    Pd4=0

        #print("After corrections:")
        #print(Pd0,Pd1,Pd2)

        print("-----------------")
        PCYdec0[i][Pd0]=1
        PCYdec1[i][Pd1]=1
        PCYdec2[i][Pd2]=1
        #PCYdec3[i][Pd3]=1
        #PCYdec4[i][Pd4]=1



    x_ax = range(len(testX))
    plt.scatter(x_ax, c0_t,  s=6, label="Err_c0",color="blue")
    plt.scatter(x_ax, c1_t,  s=6, label="Err_c1",color="red")
    plt.scatter(x_ax, c2_t,  s=6, label="Err_c2",color="green")
    #plt.scatter(x_ax, c3_t,  s=6, label="Err_c3",color="purple")
    #plt.scatter(x_ax, c4_t,  s=6, label="Err_c4",color="black")
    plt.legend()
    plt.show()
    


    print(CYdec0)

    # Confusion matrix definition por each ci, i=0,1,2,3,4

    cm0 = confusion_matrix(np.argmax(CYdec0, axis=1), np.argmax(PCYdec0, axis=1))

    cm1 = confusion_matrix(np.argmax(CYdec1, axis=1), np.argmax(PCYdec1, axis=1))

    cm2 = confusion_matrix(np.argmax(CYdec2, axis=1), np.argmax(PCYdec2, axis=1))

    #cm3 = confusion_matrix(np.argmax(CYdec3, axis=1), np.argmax(PCYdec3, axis=1))

#    cm4 = confusion_matrix(np.argmax(CYdec4, axis=1), np.argmax(PCYdec4, axis=1))
    

    n_class=np.arange(0,csize)
    n_classp=[0]*len(n_class)
    for i in range(len(n_class)):
        n_classp[i]=str(n_class[i])

    mu=0
    mu1=1
    mu2=2
    #mu3=3
    #mu4=4

    plot_confusion_matrix(mu,cm0, n_classp)
    plot_confusion_matrix(mu1,cm1, n_classp)
    plot_confusion_matrix(mu2,cm2, n_classp)
    #plot_confusion_matrix(mu3,cm3, n_classp)
    #plot_confusion_matrix(mu4,cm4, n_classp)


    # Multivariative ROC curve; 


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test=testY
    y_score=predY
    n_classes=5
    
    
    fpr, tpr, _ = roc_curve(tec0,pec0 )
    roc_auc = auc(fpr, tpr)
    
    print(roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    plt.savefig("ROCcurve20.pdf")
    plt.show()


    summarize_diagnostics(history)








from time import sleep
from multiprocessing.pool import ThreadPool

if __name__ == "__main__":

    run_test_harness()

