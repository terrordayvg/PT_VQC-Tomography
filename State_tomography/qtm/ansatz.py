import qiskit
import math
import qtm.base
import qtm.state
import numpy as np
import random
def u_onequbit_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, wire: int = 0):
    """Return a simple series of 1 qubit gate

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - wire (int): position that the gate carries on

    Returns:
        - QuantumCircuit: The circuit which have added gates
    """
    if isinstance(wire, int) != True:
        wire = (wire['wire'])
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    return qc


def u_onequbith_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, wire: int):
    """Return a simple series of 1 qubit - gate which is measured in X-basis

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - wire (int): position that the gate carries on   

    Returns:
        - QuantumCircuit: The circuit which have added gates
    """
    if isinstance(wire, int) != True:
        wire = (wire['wire'])
    qc.rz(thetas[0], wire)
    qc.rx(thetas[1], wire)
    qc.rz(thetas[2], wire)
    qc.h(wire)
    return qc


###########################
######## GHZ State ########
###########################

def create_graph_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create graph ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - (qiskit.QuantumCircuit): init circuit
    """
    n = qc.num_qubits
    edges = qtm.constant.edges_graph_state[n]
    i = 0
    for edge in edges:
        control_bit = int(edge.split('-')[0])
        controlled_bit = int(edge.split('-')[1])
        qc.crz(thetas[i], control_bit, controlled_bit)
        i += 1
    return qc


def create_stargraph_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    """Create star graph ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit: init circuit
    """
    n = qc.num_qubits
    if len(thetas) != num_layers*(2*n - 2):
        raise ValueError(
            'The number of parameter must be num_layers * (2 * n - 2)')

    j = 0
    for l in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(2, n):
            qc.ry(thetas[j], 0)
            j += 1
            qc.cz(0, i)
    return qc


def create_polygongraph_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    """Create graph ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit: init circuit
    """

    n = qc.num_qubits
    if len(thetas) != num_layers*(2*n):
        raise ValueError(
            'The number of parameter must be num_layers*(2*n)')

    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(0, n - 1, 2):
            qc.cz(i, i + 1)
        if n % 2 == 1:
            for i in range(0, n - 1):
                qc.ry(thetas[j], i)
                j += 1
        else:
            for i in range(0, n):
                qc.ry(thetas[j], i)
                j += 1
        for i in range(1, n - 1, 2):
            qc.cz(i, i + 1)
        if n % 2 == 1:
            qc.ry(thetas[j], n - 1)
            j += 1
        qc.cz(0, n - 1)
        qc.barrier()
    return qc
def create_hadamard_hypergraph_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    n = qc.num_qubits
    if len(thetas) != num_layers*(3*n):
        raise ValueError(
            'The number of parameter must be num_layers*(3*n)')
    j = 0
    for i in range(0, n):
        qc.h(i)
    for _ in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, n - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, n - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], n - 1)
        j += 1
        qc.cz(n - 2, n - 1)
        for i in range(1, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0,1,2)
    return qc

def create_hypergraph_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    n = qc.num_qubits
    if len(thetas) != num_layers*(3*n):
        raise ValueError(
            'The number of parameter must be num_layers*(3*n)')
    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, n - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, n - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], n - 1)
        j += 1
        qc.cz(n - 2, n - 1)
        for i in range(1, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.ccz(0,1,2)
    return qc

def create_hypergraph_layered_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int):
    n = qc.num_qubits
    if len(thetas) != num_layers*(6*n):
        raise ValueError(
            'The number of parameter must be num_layers*(3*n)')
    j = 0
    for _ in range(0, num_layers, 1):
        for i in range(0, n):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, 1)
        for i in range(0, n - 1):
            qc.ry(thetas[j], i)
            j += 1
        qc.cz(0, n - 1)
        qc.ry(thetas[j], 0)
        j += 1
        qc.ry(thetas[j], n - 1)
        j += 1
        qc.cz(n - 2, n - 1)
        for i in range(1, n):
            qc.ry(thetas[j], i)
            j += 1
        #qc.ccz(0,1,2)
        cnz = qiskit.circuit.library.MCMT('cz',n-1,1)
        qc.compose(cnz,qubits=list(range(0,n)),inplace=True)
        qc = create_rz_nqubit(qc, thetas[j:j + n])
        j += n
        qc = create_rx_nqubit(qc, thetas[j:j + n])
        j += n
        qc = create_rz_nqubit(qc, thetas[j:j + n])
        j += n
    return qc
# def create_GHZchecker_graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray, theta: float):
#     """Create circuit includes linear and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """

#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     # |psi_gen> = U_gen|000...>
#     qc = create_graph_ansatz(qc, thetas)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


# def create_GHZchecker_polygongraph(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
#     """Create circuit includes linear and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     # |psi_gen> = U_gen|000...>
#     qc = create_polygongraph_ansatz(qc, thetas, num_layers)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


# def create_GHZchecker_star2graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int, theta: float):
#     """Create circuit includes linear and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     qc = create_stargraph_ansatz(qc, thetas, num_layers)
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


# def create_GHZchecker_linear(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                              num_layers: int, theta: float):
#     """Create circuit includes linear and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']
#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     # |psi_gen> = U_gen|000...>
#     qc = create_linear_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


# def create_GHZchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                             num_layers: int, theta: float):
#     """Create circuit includes binho and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']
#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     # |psi_gen> = U_gen|000...>
#     qc = create_binho_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


# def create_GHZchecker_alternating_layered(qc: qiskit.QuantumCircuit,
#                                           thetas: np.ndarray, num_layers: int,
#                                           theta: float):
#     """Create circuit includes alternating layered and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']
#     if isinstance(theta, float) != True:
#         theta = theta['theta']
#     # |psi_gen> = U_gen|000...>
#     qc = create_alternating_layered_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_ghz_state_inverse(qc, theta)
#     return qc


###########################
######### W State #########
###########################



# def create_Wchecker_polygongraph(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                                  num_layers: int):
#     """Create circuit includes linear and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_polygongraph_ansatz(qc, thetas, num_layers)
#     # U_target^t|psi_gen> with U_target is W state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


# def create_Wchecker_star2graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                                num_layers: int):
#     """Create circuit includes linear and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_stargraph_ansatz(qc, thetas, num_layers)
#     # U_target^t|psi_gen> with U_target is W state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


# def create_Wchecker_graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
#     """Create circuit includes linear and GHZ

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general GHZ

#     Returns:
#         - qiskit.QuantumCircuit
#     """

#     # |psi_gen> = U_gen|000...>
#     qc = create_graph_ansatz(qc, thetas)
#     # U_target^t|psi_gen> with U_target is GHZ state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


# def create_Wchecker_linear(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                            num_layers: int):
#     """Create circuit includes linear and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_linear_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is W state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


# def create_Wchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                           num_layers: int):
#     """Create circuit includes binho and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_binho_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is W state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


# def create_Wchecker_alternating_layered(qc: qiskit.QuantumCircuit,
#                                         thetas: np.ndarray, num_layers: int):
#     """Create circuit includes Alternating layered and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_alternating_layered_ansatz(qc, thetas, num_layers=num_layers)
#     # U_target^t|psi_gen> with U_target is W state
#     qc = qtm.state.create_w_state_inverse(qc)
#     return qc


###########################
######## Haar State #######
###########################


def u_nqubit_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create enhanced version of u_1qubit ansatz

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (Numpy array): parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    j = 0
    for i in range(0, qc.num_qubits):
        qc = u_onequbit_ansatz(qc, thetas[j:j + 3], i)
        j = j + 3
    return qc


def entangle_nqubit(qc: qiskit.QuantumCircuit):
    """Create entanglement state

    Args:
        - qc (QuantumCircuit): init circuit

    Returns:
        - QuantumCircuit: the added circuit
    """
    for i in range(0, qc.num_qubits):
        if i == qc.num_qubits - 1:
            qc.cnot(qc.num_qubits - 1, 0)
        else:
            qc.cnot(i, i + 1)
    return qc


def u_cluster_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create a complicated u gate multi-qubit

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (Numpy array): parameters

    Returns:
        - QuantumCircuit: the added circuit
    """

    qc = entangle_nqubit(qc)
    qc = u_nqubit_ansatz(qc, thetas[0:qc.num_qubits * 3])
    return qc


def u_cluster_nlayer_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers):
    """Create a complicated u gate multi-qubit

    Args:
        - qc (QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - QuantumCircuit: the added circuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    params_per_layer = int(len(thetas) / num_layers)
    for i in range(0, num_layers):
        qc = entangle_nqubit(qc)
        qc = (qc, thetas[i * params_per_layer:(i + 1) * params_per_layer])
    return qc


def create_rx_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Add a R_X layer

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits):
        qc.rx(thetas[i], i)

    return qc


def create_rz_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Add a R_Z layer

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits):
        qc.rz(thetas[i], i)
    return qc


def create_cry_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create control Control-RY state

     Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    for i in range(1, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[qc.num_qubits - 1], qc.num_qubits - 1, 0)
    return qc


def create_cry_nqubit_inverse(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create control Control-RY state but swap control and target qubit

     Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    for i in range(1, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    qc.cry(thetas[qc.num_qubits - 1], 0, qc.num_qubits - 1)
    return qc


def create_linear_ansatz(qc: qiskit.QuantumCircuit,
                        thetas: np.ndarray,
                        num_layers: int = 1):
    """Create linear ansatz. The number of param is num_layers * n * 5

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * n * 5:
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i * n * 5:(i + 1) * n * 5]
        qc = create_rx_nqubit(qc, phis[:n])
        qc = create_cry_nqubit_inverse(qc, phis[n:n * 2])
        qc = create_rz_nqubit(qc, phis[n * 2:n * 3])
        qc = create_cry_nqubit(qc, phis[n * 3:n * 4])
        qc = create_rz_nqubit(qc, phis[n * 4:n * 5])
    return qc


def create_haarchecker_linear(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
                              num_layers: int, encoder):
    """Create circuit includes haar and linear

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): num_layers for linear
        - encoder: encoder for haar

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(encoder, qtm.encoding.Encoding) != True:
        encoder = encoder['encoder']
    qc1 = qiskit.QuantumCircuit(encoder.quantum_data)
    qc1 = create_linear_ansatz(qc1, thetas, num_layers=num_layers)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1


def create_haarchecker_graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray, encoder):
    """Create circuit includes haar and linear

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): num_layers for linear
        - encoder: encoder for haar

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(encoder, qtm.encoding.Encoding) != True:
        encoder = encoder['encoder']
    qc1 = qiskit.QuantumCircuit(encoder.quantum_data)
    qc1 = create_graph_ansatz(qc1, thetas)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1


def create_haarchecker_binho(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
                             num_layers: int, encoder):
    """Create circuit includes haar and linear

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): num_layers for linear
        - encoder: encoder for haar

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(encoder, qtm.encoding.Encoding) != True:
        encoder = encoder['encoder']
    qc1 = qiskit.QuantumCircuit(encoder.quantum_data)
    qc1 = create_binho_ansatz(qc1, thetas, num_layers=num_layers)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1


def create_haarchecker_alternating_layered(qc: qiskit.QuantumCircuit,
                                           thetas: np.ndarray, num_layers: int,
                                           encoder):
    """Create circuit includes Alternating layered and linear

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): num_layers for linear
        - encoder: encoder for haar

    Returns:
        - qiskit.QuantumCircuit
    """
    if isinstance(num_layers, int) != True:
        num_layers = num_layers['num_layers']
    if isinstance(encoder, qtm.encoding.Encoding) != True:
        encoder = encoder['encoder']
    qc1 = qiskit.QuantumCircuit(encoder.quantum_data)
    qc1 = create_alternating_layered_ansatz(qc1, thetas, num_layers=num_layers)
    qc1 = qc1.combine(qc.inverse())
    qc1.add_register(qiskit.ClassicalRegister(encoder.num_qubits))
    return qc1


###########################
######## Binho State ######
###########################


def create_wy_ansatz_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create WY state

     Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (Numpy array): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    for i in range(1, qc.num_qubits - 1, 2):
        qc.cry(thetas[i], i + 1, i)
    qc.cry(thetas[qc.num_qubits - 1], 0, qc.num_qubits - 1)
    return qc


def create_binho_ansatz(qc: qiskit.QuantumCircuit, thetas: np.ndarray, num_layers: int = 1):
    """Create linear ansatz 

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (Int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * n * 5:
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i * n * 5:(i + 1) * n * 5]
        qc = create_rx_nqubit(qc, phis[:n])
        qc = create_wy_ansatz(qc, phis[n:n * 2])
        qc = create_rz_nqubit(qc, phis[n * 2:n * 3])
        qc = create_wy_ansatz(qc, phis[n * 3:n * 4])
        qc = create_rz_nqubit(qc, phis[n * 4:n * 5])
    return qc


###########################
###### Layered State ######
###########################


def create_ry_nqubit(qc: qiskit.QuantumCircuit, thetas: np.ndarray, shift=0):
    """Add a R_Y layer

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - shift (int): start index
    Returns:
        - qiskit.QuantumCircuit
    """
    if qc.num_qubits - shift < len(thetas):
        raise Exception(
            'Number of parameters must be equal num_qubits - shift')

    for i in range(0, len(thetas)):
        qc.ry(thetas[i], i + shift)
    return qc


def create_swap_nqubit(qc: qiskit.QuantumCircuit, shift=0):
    """Add a SWAP layer

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - shift (Int): start index
    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0 + shift, qc.num_qubits - 1, 2):
        qc.swap(i, i + 1)
    return qc


def create_alternating_layered_ansatz(qc: qiskit.QuantumCircuit,
                                    thetas: np.ndarray,
                                    num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    if len(thetas) != num_layers * (n * 5 - 4):
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 5')
    for i in range(0, num_layers):
        phis = thetas[i * (n * 5 - 4):(i + 1) * (n * 5 - 4)]
        qc.barrier()
        qc = create_ry_nqubit(qc, phis[:n])
        qc = create_swap_nqubit(qc)
        qc = create_ry_nqubit(qc, phis[n:n * 2 - 1])
        qc = create_swap_nqubit(qc, shift=1)
        qc = create_ry_nqubit(qc, phis[n * 2 - 1:n * 3 - 2], shift=1)
        qc = create_swap_nqubit(qc)
        qc = create_ry_nqubit(qc, phis[n * 3 - 2:n * 4 - 3])
        qc = create_swap_nqubit(qc, shift=1)
        qc = create_ry_nqubit(qc, phis[n * 4 - 3:n * 5 - 4], shift=1)
    return qc


###########################
#### Tomography circuit ###
###########################
from qiskit.circuit.library import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
def random_circuit(num_qubits, depth,thetas, max_operands=3, measure=False,
                   conditional=False, reset=False, seed=None):
    
    if max_operands < 1 or max_operands > 3:
        raise CircuitError("max_operands must be between 1 and 3")




    one_q_ops = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate, HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    one_param = [RXGate, RYGate, RZGate,CRZGate,CRYGate,CRXGate,CU1Gate,U1Gate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]
    two_q_ops = [CXGate,CRZGate,CRYGate,CRXGate]
    three_q_ops = [CCXGate, CSwapGate]

    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, 'c')
        qc.add_register(cr)

    if reset:
        one_q_ops += [Reset]

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)
    counter=0
    # apply arbitrary random operations at every depth
    for ni in range(depth):
        # choose either 1, 2, or 3 qubits for the operation
        remaining_qubits = list(range(num_qubits))
        while remaining_qubits:
            
            max_possible_operands = min(len(remaining_qubits), max_operands)
            num_operands = rng.choice(range(max_possible_operands)) + 1
            rng.shuffle(remaining_qubits)
            operands = remaining_qubits[:num_operands]
            remaining_qubits = [q for q in remaining_qubits if q not in operands]
            if num_operands == 1:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3:
                operation = rng.choice(three_q_ops)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            # [thetas[counter]]
            
            register_operands = [qr[i] for i in operands]
            angles = [rng.uniform(0, 2 * np.pi) for x in range(num_angles)]
            #print(angles[0])
            #print(type(*angles))
            #A=*angles
            #print(type(A))
            print(operation)
            
            if not angles:
                op = operation()
            else: 
                op = operation(*angles)

            counter=counter+1

            # with some low probability, condition on classical bit values
            if conditional and rng.choice(range(10)) == 0:
                value = rng.integers(0, np.power(2, num_qubits))
                op.condition = (cr, value)

            qc.append(op, register_operands)

    if measure:
        qc.measure(qr, cr)

    return qc

def create_Wchain(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1):
        qc.cry(thetas[i], i, i + 1)
    qc.cry(thetas[-1], qc.num_qubits - 1, 0)
    return qc


def create_WchainCNOT(qc: qiskit.QuantumCircuit):
    """Create W_chain ansatz but replacing CRY gate by CNOT gate

    Args:
        - qc (qiskit.QuantumCircuit): init circuit

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1):
        qc.cnot(i, i + 1)
    qc.cnot(qc.num_qubits - 1, 0)
    return qc


def create_Walternating(qc: qiskit.QuantumCircuit, thetas: np.ndarray, index_layer):
    """Create W_alternating ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - index_layer (int)
    Returns:
        - qiskit.QuantumCircuit
    """
    t = 0
    if index_layer % 2 == 0:
        # Even
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
        qc.cry(thetas[-1], 0, qc.num_qubits - 1)
    else:
        # Odd
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cry(thetas[t], i, i + 1)
            t += 1
    return qc


def create_WalternatingCNOT(qc: qiskit.QuantumCircuit, index_layer):
    """Create Walternating CNOT

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - index_layer (int): limit layer

    Returns:
        - qiskit.QuantumCircuit
    """
    t = 0
    if index_layer % 2 == 0:
        for i in range(1, qc.num_qubits - 1, 2):
            qc.cnot(i, i + 1)
            t += 1
        qc.cnot(0, qc.num_qubits - 1)
    else:
        for i in range(0, qc.num_qubits - 1, 2):
            qc.cnot(i, i + 1)
            t += 1
    return qc


def create_Walltoall(qc: qiskit.QuantumCircuit, thetas: np.ndarray, limit=0):
    """Create Walltoall

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - limit (int): limit layer

    Returns:
        - qiskit.QuantumCircuit
    """
    if limit == 0:
        limit = len(thetas)
    t = 0
    for i in range(0, qc.num_qubits):
        for j in range(i + 1, qc.num_qubits):
            qc.cry(thetas[t], i, j)
            t += 1
            if t == limit:
                return qc
    return qc


def create_WalltoallCNOT(qc: qiskit.QuantumCircuit, limit=0):
    """Create Walltoall CNOT

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - limit (int): limit layer

    Returns:
        - qiskit.QuantumCircuit
    """
    t = 0
    for i in range(0, qc.num_qubits):
        for j in range(i + 1, qc.num_qubits):
            qc.cnot(i, j)
            t += 1
            if t == limit:
                return qc
    return qc


def create_Wchain_layered_ansatz(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n * 2):(i + 1) * (n * 2)]
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wchain(qc, phis[:n])
        qc.barrier()
        qc = create_rx_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])

    return qc


def create_Wchain_layered_ansatz2(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    print(thetas)

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n * 3):(i + 1) * (n * 3)]
        #print("Phis:")
        #print(phis)
        #print("")
        print(phis)
        qc = create_Wchain(qc, phis[:n])
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[n:n * 2])
        qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])

       
    return qc


def create_Wfirst(qc: qiskit.QuantumCircuit, thetas: np.ndarray,thetas2: np.ndarray,thetas3: np.ndarray):
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    ########################

    #qtm.ansatz.random_circuit(2,3,thetas,measure=False,seed=2)
    #qc.ry(thetas[0], 0)
    
    qc.cry(thetas[0],0,1)
    #qc.cnot(0,1)
    qc.cry(thetas[1], 1,2)

    qc.cry(thetas[2], 2,3)

    qc.cry(thetas[3], 3,4)

    qc.cry(thetas[4], 4,0)
    
    qc.rz(thetas2[0], 0)
    qc.rx(thetas3[0], 0)

    qc.rz(thetas2[1], 1)
    qc.rx(thetas3[1], 1)

    qc.rz(thetas2[2], 2)
    qc.rx(thetas3[2], 2)

    qc.rz(thetas2[3], 3)
    qc.rx(thetas3[3], 3)

    qc.rz(thetas2[4], 4)
    qc.rx(thetas3[4], 4)


    #qc.cnot(0,1)
    #qc.rx(thetas3[0], 0)
    
    #qc.barrier()
    ##########################
    #qc.ry(thetas[1], 1)
    
    
    
    

    #qc.cnot(qc.num_qubits - 1,0)
    #qc.rz(thetas3[2], 2)

    ###############################

    #qc.crx(thetas[0], 0, 0 + 1)

    #for i in range(0, qc.num_qubits - 1):
    #    qc.crx(thetas[i], i, i + 1)
    #qc.cry(thetas[-1], qc.num_qubits - 1, 0)
    return qc


#### This is the current solution that best worksout with this - to cut out the z gate still the learning rate is pretty good in variational circuit
def create_random_ansatz(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n *3):(i + 1) * (n * 3)]
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wfirst(qc, phis[:n], phis[n:n * 2],phis[n*2:n * 3])

        #qc.barrier()
        #qc = create_rz_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])
        #print(qc)
       
    return qc


def create_Wfirst2(qc: qiskit.QuantumCircuit, thetas: np.ndarray,thetas2: np.ndarray):
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    ########################

    #qtm.ansatz.random_circuit(2,3,thetas,measure=False,seed=2)
    #qc.ry(thetas[0], 0)
        #qc.ry(thetas2[0], 0)

    
    qc.rx(thetas[0], 0)
    #qc.cnot(0,1)
    qc.rx(thetas[1], 1)
    #qc.rx(thetas[2], 2)

    qc.cnot(0,1)
    #qc.cnot(2,3)

    #qc.cnot(1,0)

    qc.ry(thetas2[0], 0)
    qc.ry(thetas2[1], 1)
    #qc.ry(thetas2[2], 2)

    #qc.cnot(1,2)
    

   # qc.cry(thetas3[1], 2,0)
   # qc.ry(thetas3[0], 0)
   # qc.rx(thetas3[2], 0)



    #qc.rz(thetas4[0], 0)
    #qc.cnot(0,1)
    #qc.cry(thetas[1], 1,2)

    #qc.rx(thetas2[1], 1)
    #qc.ry(thetas3[1], 1)
    #qc.rz(thetas4[1], 1)


    #qc.crz(thetas[2], 2,0)
    

    
    #qc.rz(thetas4[2], 2)


    return qc


def create_Wfirst3(qc: qiskit.QuantumCircuit, thetas: np.ndarray,thetas2: np.ndarray,thetas3: np.ndarray,thetas4: np.ndarray):
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    ########################

    #qtm.ansatz.random_circuit(2,3,thetas,measure=False,seed=2)
    #qc.ry(thetas[0], 0)
        #qc.ry(thetas2[0], 0)

    
    qc.cry(thetas[0], 0,1)
    #qc.cnot(0,1)
    #qc.cry(thetas[1], 1,2)
    qc.cry(thetas[1], 2,3)
    #qc.cry(thetas[3], 3,0)

    qc.rx(thetas2[0], 0)
    qc.rx(thetas2[1], 1)
    qc.rx(thetas2[2], 2)
    qc.rx(thetas2[3], 3)

    qc.cry(thetas3[0], 1,2)
    qc.cry(thetas3[1], 3,0)

    qc.ry(thetas4[0], 0)
    qc.ry(thetas4[1], 1)
    qc.ry(thetas4[2], 2)
    qc.ry(thetas4[3], 3)




    return qc



#### This is the current solution that best worksout with this - to cut out the z gate still the learning rate is pretty good in variational circuit
def create_cx_ansatz(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wfirst2(qc, phis[n*2*i:n*2*i+n], phis[n*2*i+n:n*2*i+2*n])

        #qc.barrier()
        #qc = create_rz_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])
        #print(qc)
    
    #Last layer
    thetas3=phis[len(thetas)-n:len(thetas)]
    qc.rx(thetas3[0], 0)
    #qc.cnot(0,1)
    qc.rx(thetas3[1], 1)
    #qc.rx(thetas3[2], 2)
    #qc.rx(thetas3[3], 3)
    
    return qc


def create_random_ansatz3(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wfirst3(qc, phis[:2], phis[2:6],phis[6:8],phis[8:12])

        #qc.barrier()
        #qc = create_rz_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])
        #print(qc)
       
    return qc


def create_Wchain_layered_ansatz2(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n * 3):(i + 1) * (n * 3)]
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wchain(qc, phis[:n])
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[n:n * 2])
        qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])

       
    return qc

def Normal_ansatz(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n * 4):(i + 1) * (n * 4)]
        #print("Phis:")
        #print(phis)
        #print("")
        qc = create_Wchain(qc, phis[:n])
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[n:n * 2])
        qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        qc = create_rz_nqubit(qc, phis[n * 3:n * 4])

       
    return qc

def create_Wchain_layered_ansatz3(qc: qiskit.QuantumCircuit,
                               thetas: np.ndarray,
                               num_layers: int = 1):
    """Create Alternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    #if len(thetas) != num_layers * (n * 4):
    #    raise Exception(
    #        'Number of parameters must be equal n_layers * num_qubits * 4')
    for i in range(0, num_layers):
        # 3 turn into 4
        phis = thetas[i * (n * 2):(i + 1) * (n * 2)]
        #print("Phis:")
        #print(phis)
        #print("")



        qc = create_Wchainr(qc, phis[:n])
        qc.barrier()
        qc = create_ry_nqubit(qc, phis[n:n * 2])
        #qc = create_rx_nqubit(qc, phis[n * 2:n * 3])
        #qc = create_rz_nqubit(qc, phis[n * 3:n * 4])

       
    return qc

def create_Wchainr(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    """Create W_chain ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters

    Returns:
        - qiskit.QuantumCircuit
    """
    for i in range(0, qc.num_qubits - 1):
        qc.crx(thetas[i], i, i + 1)
    qc.cry(thetas[-1], qc.num_qubits - 1, 0)
    return qc



def create_WchainCNOT_layered_ansatz(qc: qiskit.QuantumCircuit,
                                   thetas: np.ndarray,
                                   num_layers: int = 1):
    """Create WchainCNOT layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (Int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    if len(thetas) != num_layers * (n * 3):
        raise Exception(
            'Number of parameters must be equal n_layers * num_qubits * 3')
    for i in range(0, num_layers):
        phis = thetas[i * (n * 3):(i + 1) * (n * 3)]
        qc = create_WchainCNOT(qc)
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[:n])
        qc = create_rx_nqubit(qc, phis[n:n * 2])
        qc = create_rz_nqubit(qc, phis[n * 2:n * 3])
    return qc


def calculate_n_walternating(index_layers, num_qubits):
    if index_layers % 2 == 0:
        n_walternating = int(num_qubits / 2)
    else:
        n_walternating = math.ceil(num_qubits / 2)

    return n_walternating


def create_Walternating_layered_ansatz(qc: qiskit.QuantumCircuit,
                                     thetas: np.ndarray,
                                     num_layers: int = 1):
    """Create Walternating layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (Int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    n_param = 0
    for i in range(0, num_layers):
        n_alternating = qtm.ansatz.calculate_n_walternating(i, n)
        phis = thetas[n_param:n_param + n_alternating + 3 * n]
        n_param += n_alternating + 3 * n
        qc = create_Walternating(qc, phis[:n_alternating], i + 1)
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[n_alternating:n_alternating + n])
        qc = create_rx_nqubit(
            qc, phis[n_alternating + n:n_alternating + n * 2])
        qc = create_rz_nqubit(
            qc, phis[n_alternating + n * 2:n_alternating + n * 3])
    return qc


def create_WalternatingCNOT_layered_ansatz(qc: qiskit.QuantumCircuit,
                                         thetas: np.ndarray,
                                         num_layers: int = 1):
    """Create WalternatingCNOT layered ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - n_layers (Int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits

    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    for i in range(0, num_layers):
        phis = thetas[i * (n * 3):(i + 1) * (n * 3)]
        qc = create_WalternatingCNOT(qc, i + 1)
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[:n])
        qc = create_rx_nqubit(qc, phis[n: n * 2])
        qc = create_rz_nqubit(qc, phis[n * 2: n * 3])
    return qc


def calculate_n_walltoall(n):
    n_walltoall = 0
    for i in range(1, n):
        n_walltoall += i
    return n_walltoall


def create_Walltoall_layered_ansatz(qc: qiskit.QuantumCircuit,
                                  thetas: np.ndarray,
                                  num_layers: int = 1,
                                  limit=0):
    """Create W all to all ansatz

    Args:
        - qc (qiskit.QuantumCircuit): init circuit
        - thetas (np.ndarray): parameters
        - num_layers (int): numpy of layers

    Returns:
        - qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    n_walltoall = calculate_n_walltoall(n)
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])

    if len(thetas) != num_layers * (3 * n) + num_layers * n_walltoall:
        raise Exception(
            'Number of parameters must be equal num_layers*(3*n) + num_layers*n_walltoall'
        )
    for i in range(0, num_layers):
        phis = thetas[i * (3 * n) + i * n_walltoall:(i + 1) * (3 * n) +
                      (i + 1) * n_walltoall]
        qc = create_Walltoall(qc, phis[0:n_walltoall], limit=limit)
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[n_walltoall:n_walltoall + n])
        qc.barrier()
        qc = create_rx_nqubit(qc, phis[n_walltoall + n:n_walltoall + n * 2])
        qc.barrier()
        qc = create_rz_nqubit(qc,
                              phis[n_walltoall + n * 2:n_walltoall + n * 3])
        qc.barrier()
    return qc


def create_WalltoallCNOT_layered_ansatz(qc: qiskit.QuantumCircuit,
                                      thetas: np.ndarray,
                                      num_layers: int = 1,
                                      limit=0):
    """Create W all to all ansatz

    Args:
        - qc (qiskit.QuantumCircuit): Init circuit
        - thetas (np.ndarray): Parameters
        - num_layers (Int): numpy of layers

    Returns:
        qiskit.QuantumCircuit
    """
    n = qc.num_qubits
    if isinstance(num_layers, int) != True:
        num_layers = (num_layers['num_layers'])
    for i in range(0, num_layers):
        phis = thetas[i * (3 * n):(i + 1) * (3 * n)]
        qc = create_WalltoallCNOT(qc, limit=limit)
        qc.barrier()
        qc = create_rz_nqubit(qc, phis[: n])
        qc = create_rx_nqubit(qc, phis[n: n * 2])
        qc = create_rz_nqubit(qc, phis[n * 2: n * 3])
    return qc


# def create_AMEchecker_polygongraph(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                                  num_layers: int):
#     """Create circuit includes linear and W

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general AME

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_polygongraph_ansatz(qc, thetas, num_layers)
#     # U_target^t|psi_gen> with U_target is AME state
#     qc = qtm.state.create_AME_state_inverse(qc)
#     return qc

# def create_AMEchecker_star2graph(qc: qiskit.QuantumCircuit, thetas: np.ndarray,
#                                num_layers: int):
#     """Create circuit includes linear and AME

#     Args:
#         - qc (qiskit.QuantumCircuit): init circuit
#         - thetas (np.ndarray): parameters
#         - num_layers (int)
#         - theta (float): param for general W

#     Returns:
#         - qiskit.QuantumCircuit
#     """
#     if isinstance(num_layers, int) != True:
#         num_layers = num_layers['num_layers']

#     # |psi_gen> = U_gen|000...>
#     qc = create_stargraph_ansatz(qc, thetas, num_layers)
#     # U_target^t|psi_gen> with U_target is AME ßstate
#     qc = qtm.state.create_AME_state_inverse(qc)
#     return qc


def random_ccz_circuit(num_qubits, num_gates):
    """Adds a random number of CZ or CCZ gates (up to `max_gates`) to the given circuit."""
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    sure = True if num_gates < 3 else False
    for _ in range(num_gates):
        if np.random.randint(2, size=1) == 0 or sure:
            wires = random.sample(range(0, num_qubits), 2)
            qc.cz(wires[0], wires[1])
        else:
            wires = random.sample(range(0, num_qubits), 3)
            wires.sort()
            qc.ccz(wires[0], wires[1], wires[2])
    return qc

def ry_layer(num_qubits, num_layers, thetas):
    """Adds a random number of CZ or CCZ gates (up to `max_gates`) to the given circuit."""
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for j in range(0, num_layers):
        for i in range(num_qubits):
            qc.ry(thetas[i+j*num_qubits], i)
    return qc

def g2(qc: qiskit.QuantumCircuit, thetas: np.ndarray):
    num_qubits = qc.num_qubits
    j = 0
    for i in range(num_qubits):
        qc.ry(thetas[j], i)
        j += 1
    for i in range(0, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    for i in range(num_qubits):
        qc.ry(thetas[j], i)
        j += 1
    for i in range(1, num_qubits - 1, 2):
        qc.cz(i, i + 1)
    qc.cz(0, num_qubits - 1)
    return qc