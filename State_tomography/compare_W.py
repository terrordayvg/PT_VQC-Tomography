import qiskit, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(1, '../')
import qtm.qcompilation
from qiskit.quantum_info import random_unitary
from qiskit.extensions import UnitaryGate



#np.random.seed(42)
#num_qubits = 4
#num_layers = 1
#thetas = np.ones(num_layers*num_qubits*2)
#psi = 2*np.random.rand(2**num_qubits)-1
#psi = psi / np.linalg.norm(psi)
#encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')

#qc = qiskit.QuantumCircuit(num_qubits)
#qc = qtm.ansatz.create_Wchain_layered_ansatz(qc, thetas, num_layers=num_layers)
#print(qc)

#num_qubits = 4
#num_layers = 1
#thetas = np.ones(num_layers*num_qubits*2)
#psi = 2*np.random.rand(2**num_qubits)-1
#psi = psi / np.linalg.norm(psi)
#u = qiskit.QuantumCircuit(num_qubits, num_qubits)
#u.initialize(psi, range(0, num_qubits))

#u=qtm.ansatz.random_circuit(3,200,thetas,measure=False,seed=2)

#print(u)
#
#compiler = qtm.qcompilation.QuantumCompilation(u=u,
#        vdagger = qtm.ansatz.create_Wchain_layered_ansatz,
#        optimizer = 'adam',
#        loss_func = 'loss_basic',
#        thetas = thetas,
#        num_layers = num_layers
#    )

#compiler.fit(num_steps=30,verbose=1)
#plt.plot(compiler.loss_values,label='CRy_Rx')

#This function is essential as it communicates with the adapt_pol.py to have exactly the same unitary
def Create_Haar_U(num_qubits,seed1):
  CHar=random_unitary(2**num_qubits, seed=seed1)
  return CHar



def Run_Vd_method(Num,Max_Tr,num_qubits,num_layers):
############################################### Full on 
  print("------- Initialize Vd method ----------")  
  Mean=[0]*Num
  Mean2=[0]*Num


  for trig in range(Max_Tr):
  #np.random.seed(trig+1000)
    thetas = np.ones(num_layers*num_qubits*2)
  #psi = 2*np.random.rand(2**num_qubits)-1
  #psi = psi / np.linalg.norm(psi)
   

  #If you truly want to apply a haar random unitary 
    u = qiskit.QuantumCircuit(num_qubits, num_qubits)
    CHar=Create_Haar_U(num_qubits,42)
  #CHar=random_unitary(2**num_qubits, seed=trig+1512)
    print("")
    CH=UnitaryGate(CHar.data,label="Har")
    print("Matrix")
    print("")
    print(CHar.data)

    u.append(CH, [0,1,2])

  #u = qiskit.QuantumCircuit(num_qubits, num_qubits)
  #u.initialize([1/np.sqrt(2),1/np.sqrt(2)],0)
  #u.initialize([1/np.sqrt(2),1/np.sqrt(2)],1)

    print(u)


    thetas = np.ones(num_layers*num_qubits*2+num_qubits)
    print(thetas)
    A = qiskit.QuantumCircuit(num_qubits)
  #print(qtm.ansatz.create_random_ansatz2(A, thetas, num_layers=num_layers))

  #A = qiskit.QuantumCircuit(num_qubits)
  #print(qtm.ansatz.create_random_ansatz(A, thetas, num_layers=num_layers))
#print(u)

    compiler3 = qtm.qcompilation.QuantumCompilation(u=u,
        vdagger = qtm.ansatz.create_cx_ansatz,    #create_Wchain_layered_ansatz3
        optimizer = 'adam',
        loss_func = 'loss_basic',
        thetas = thetas,
        num_layers = num_layers
    )

    compiler3.fit(num_steps=Num,verbose=1)


    targeT=compiler3.loss_values
 

  #u.initialize(psi, range(0, num_qubits))




## Model with missing rz at the end --------------------------------------------------------------------

    thetas = np.ones(num_layers*num_qubits*3)
    A = qiskit.QuantumCircuit(num_qubits)
    print(qtm.ansatz.create_Wchain_layered_ansatz2(A, thetas, num_layers=num_layers))

    compiler2 = qtm.qcompilation.QuantumCompilation(u=u,
        vdagger = qtm.ansatz.create_Wchain_layered_ansatz2,
        optimizer = 'adam',
        loss_func = 'loss_basic',
        thetas = thetas,
        num_layers = num_layers
    )

    compiler2.fit(num_steps=Num,verbose=1)
  #plt.plot(compiler2.loss_values,label='CRy_RzRx')
#print("Loss values")

#print(compiler2.loss_values)
############################################### CRX Ry

    origiN=compiler2.loss_values

  
    #plt.plot(origiN,label='cry_chain_rzrx_Adam'+str(trig))

  #num_qubits = 4
  #num_layers = 2
  #thetas = np.ones(num_layers*num_qubits*3)
  #print(" ")
  #A = qiskit.QuantumCircuit(num_qubits)
  #print(qtm.ansatz.qtm.ansatz.create_random_ansatz3(A, thetas, num_layers=num_layers))

  #compiler4 = qtm.qcompilation.QuantumCompilation(u=u,
   #     vdagger = qtm.ansatz.create_random_ansatz2,
  #      optimizer = 'qng',
   #     loss_func = 'loss_basic',
  #      thetas = thetas,
   #     num_layers = num_layers
  #  )

  #compiler4.fit(num_steps=Num,verbose=1)
  #plt.plot(compiler2.loss_values,label='CRy_RzRx')
#print("Loss values")

#print(compiler2.loss_values)
############################################### CRX Ry

  #New=compiler4.loss_values
  #plt.plot(New,label='cnot_model_even_oddL2QNG'+str(trig))


   # plt.plot(targeT,label="Cx"+str(trig))


    with open('C:/Users/vladl/Documents/PHD/Simulations/Qiskit-Simulations/SECURITY/Images_new_model/3qubits_d.txt', 'a') as output:

      output.write("\n")

      output.write("Cry_rzrxrz")
      output.write(str(origiN))
      output.write("\n")

      output.write("cx")
      output.write(str(targeT))
      output.write("\n")

    #output.write("New_model")
    #output.write(str(New))
      output.write("\n")

      output.write("------------------------------------")

      output.close()



    for k in range(len(targeT)):
      Ji=targeT[k]-origiN[k]
      Mean[k]=Mean[k]+Ji

    #Ki=New[k]-origiN[k]
    #Mean2[k]=Mean2[k]+Ki  




  for li in range(Num):
    Mean[li]=Mean[li]/Max_Tr
 # Mean2[li]=Mean2[li]/Max_Tr

 # plt.plot(Mean,label='Mean_Var_cx_model')
#plt.plot(Mean2,label='Mean_cry_model')

  #plt.axhline(y=0, color='r', linestyle='-')

  #plt.savefig('C:/Users/vladl/Documents/PHD/Simulations/Qiskit-Simulations/SECURITY/Images_new_model/'+'3qubits_d'+str(Num)+str(Max_Tr)+'.png', bbox_inches='tight')
  #plt.legend()
  #plt.show()

  return origiN,targeT





################### Main ####################

DataNum=30 # Number of steps
Max_Tr=1 #Number of trials
num_qubits = 3 #Number of qubits
num_layers = 1 #Number of layers


if __name__ == "__main__":
  Linear_cry,Linear_cx=Run_Vd_method(DataNum,Max_Tr,num_qubits,num_layers)