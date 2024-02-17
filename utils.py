import numpy as np
from numpy.linalg import eigh, matrix_rank
from scipy.linalg import logm
from math import log
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer, ClassicalRegister
from copy import deepcopy
import json

import TensorNetwork
import hamiltonians
import Entangler

sv_backend = Aer.get_backend("statevector_simulator")


def get_state(circ):
    job = execute(circ, sv_backend)
    result = job.result()
    state = result.get_statevector(circ)
    return state


def make_hadamard_test_circuit(circ: QuantumCircuit,
                               q_anc: QuantumRegister,
                               c_anc: ClassicalRegister) -> QuantumCircuit:
    """Takes a circuit, makes it a controlled circuit, and creates a
        Hadamard test circuit of all this stuff"""
    c_circ = make_control_circuit(circ, q_anc, c_anc)
    had_circ = QuantumCircuit()
    for reg in circ.qregs:
        had_circ.add_register(reg)
    for reg in circ.cregs:
        had_circ.add_register(reg)
    had_circ.add_register(q_anc)
    had_circ.add_register(c_anc)

    had_circ.h(q_anc[0])
    had_circ += c_circ
    had_circ.h(q_anc[0])

    return had_circ


def make_control_circuit(circ: QuantumCircuit,
                         q_anc: QuantumRegister,
                         c_anc: ClassicalRegister) -> QuantumCircuit:
    """Take circ and return a different circuit with one ancilla
    and all gates becoming controlled by that ancilla.
    For now, a quick and dirty solution is in place.
    Read at your discretion, you may get eye cancer"""
    c_circ = QuantumCircuit()
    for reg in circ.qregs:
        c_circ.add_register(reg)
    for reg in circ.cregs:
        c_circ.add_register(reg)
    c_circ.add_register(q_anc)
    c_circ.add_register(c_anc)

    for gate in circ:
        title = gate.name  # 'ry'
        ##rotations
        if title == 'rx':
            c_circ.h(gate.qargs[0])
            c_circ.crz(gate.param[0], q_anc[0], gate.qargs[0])
            c_circ.h(gate.qargs[0])
        elif title == 'ry':
            c_circ.s(gate.qargs[0])
            c_circ.h(gate.qargs[0])
            c_circ.crz(gate.param[0], q_anc[0], gate.qargs[0])
            c_circ.h(gate.qargs[0])
            c_circ.sdg(gate.qargs[0])
        elif title == 'rz':
            c_circ.crz(gate.param[0], q_anc[0], gate.qargs[0])
        ## Pauli
        elif title == 'x':
            c_circ.cx(q_anc[0], gate.qargs[0])
        elif title == 'y':
            c_circ.cy(q_anc[0], gate.qargs[0])
        elif title == 'z':
            c_circ.cz(q_anc[0], gate.qargs[0])
        ## Clifford
        elif title == 'h':
            c_circ.ch(q_anc[0], gate.qargs[0])
        elif title == 's':
            c_circ.cu1(np.pi / 2, q_anc[0], gate.qargs[0])

        elif title == 'u1':
            c_circ.cu1(gate.param[0], q_anc[0], gate.qargs[0])
        elif title == 'u2':
            c_circ.cu2(gate.param[0], gate.param[1], q_anc[0], gate.qargs[0])
        elif title == 'u3':
            c_circ.cu3(gate.param[0], gate.param[2], gate.param[2], q_anc[0], gate.qargs[0])

        elif title == 'cx':
            c_circ.ccx(q_anc[0], gate.qargs[0], gate.qargs[1])

        elif title == 'barrier':
            c_circ.barrier()

    return c_circ

def jsonize(xk):
    """Turn a numpy array of complex numbers
    into a list of pairs of floats"""
    return [[z.real, z.imag] for z in xk]


def unjsonize(state_jsoned):
    """Takes a list of pairs (x,y) and returns a list of x + iy"""
    return np.array([pair[0] + 1j * pair[1] for pair in state_jsoned])


def psi_to_rho(psi):
    """Takes vector and returns density matrix"""
    return np.outer(psi, np.array(psi).conj())
    # vec = array(v).reshape((len(v), 1))
    # #vec2 = array(v2).reshape((len(v2), 1))
    #
    # rho = vec @ vec.T.conj()
    # #rho2 = vec2 @ vec2.T.conj()
    # return rho


def state_partition(psi, m=1):
    """Partitions a state into the first m qubits and the rest,
    returns the reduced density matrix
    TOFIX: It does something wrong as entropy of parts is not the same"""
    rho = psi_to_rho(psi)
    n_qubits = round(log(np.shape(rho)[0]) / log(2))
    tensor_shape = [2] * n_qubits * 2
    rho = rho.reshape(tensor_shape)
    if m >= n_qubits or m == 0:
        raise ValueError('Invalid separation line')
    qubits_to_contract = n_qubits - m
    for i in range(qubits_to_contract):
        rho = np.trace(rho,
                       axis1=(n_qubits - 1 - i),
                       axis2=(2 * n_qubits - 2 - 2 * i))
    # pack rho back into matrix form
    return rho.reshape((2**m, 2**m))


def get_entropy(rho):
    """Calculates the von Neumann entropy of a density matrix.
    See also https://stackoverflow.com/questions/51898197/"""
    # w, v = np.linalg.eigh(rho)
    # return -np.sum(w * np.log2(w))
    e, v = np.linalg.eigh(rho)
    cutoff = 1e-12
    nonzero = np.where(e > cutoff)
    # I introduced cutoff so that very small eigenvalues don't lead to
    # singularities
    S = 0
    for i in nonzero[0]:
        S -= (e[i] * np.log(e[i]))
    return S


def get_state_rank(v, n_1=None):
    """Cuts the system in half and evaluates the rank of the reduced
    density matrix

    """
    n_qubits = int(log(len(v)) / log(2))
    if n_1 is None:
        n_1 = n_qubits // 2
    n_2 = n_qubits - n_1
    rho = psi_to_rho(v)
    tens = rho.reshape(list(2 for i in range(2 * n_qubits)))
    axis_pairs = [(0, n_2 + n_1 - i) for i in range(n_1)]
    for i in range(n_1):
        tens = np.trace(tens, axis1=axis_pairs[i][0], axis2=axis_pairs[i][1])
    rho_reduced = tens.reshape((2**n_2, 2**n_2))
    return matrix_rank(rho_reduced)


def get_reduced_rho_eigs(v, n_1=None):
    """cuts the system in half and returns the eigenvalues of the reduced DM"""
    n_qubits = int(log(len(v)) / log(2))
    if n_1 is None:
        n_1 = n_qubits // 2
    n_2 = n_qubits - n_1
    rho = psi_to_rho(v)
    tens = rho.reshape(list(2 for i in range(2 * n_qubits)))
    axis_pairs = [(0, n_2 + n_1 - i) for i in range(n_1)]
    for i in range(n_1):
        tens = np.trace(tens, axis1=axis_pairs[i][0], axis2=axis_pairs[i][1])
    rho_reduced = tens.reshape((2**n_2, 2**n_2))
    w, vec = eigh(rho_reduced)
    return w, vec


def KL_divergence(v1, v2):
    """Takes two state vectors, (turns them columns), transforms them to
    density matrices, then computes the Kullback-Leibler divergence
    between the two
    """
    rho1, rho2 = psi_to_rho(v1), psi_to_rho(v2)
    
    return np.trace(rho1 @ logm(rho1) - rho1 @ logm(rho2))


def trace_distance(v1, v2):
    rho1, rho2 = psi_to_rho(v1), psi_to_rho(v2)
    delta = rho1 - rho2
    w, v = eigh(delta)
    return 0.5 * np.sum(abs(w))


def plot_checkers_vs_tree():
    
    h = np.linspace(0, 2, num=11)
    E_tree = np.array([-5.99998422475232, -6.046833295896431, -6.217214148991607,
              -6.491184781574833, -6.811360050213664, -7.343102382693722,
              -7.870716542997718, -8.87742054242434, -9.819118415208449,
              -10.84290667718848, -11.989050818000822])

    E_checkers = np.array([-5.999984224752323, -6.054283985978191, -6.226538357286587,
                  -6.510023996127591, -6.857444358108586, -7.388619177836939,
                  -7.976269308343067, -8.909322800313337, -9.821872398776941,
                  -10.85600069802038, -11.99418562096905])

    E_exact = np.array([-6.00, -6.060167024024001, -6.243443643699888,
               -6.563001032061952, -7.048804475009548, -7.727406610312548,
               -8.577996550494136, -9.545045186933955, -10.58219667733395,
               -11.662046331906774, -12.769389127207353])

    #plt.plot(h, E_exact, label='Exact')
    plt.plot(h, (E_tree - E_exact) / abs(E_exact), 'bo', label='Tree')
    plt.plot(h, (E_checkers - E_exact) / abs(E_exact), 'ro', label='Checkers')
    plt.xlabel('h')
    plt.ylabel('(E - E_exact) / abs(E_exact)')
    plt.legend()
    plt.show()


def plot_fidelity():
    f = open('TTN6_params', 'r')
    S = f.read()
    S = S.split('\n')
    params_data = [json.loads(S[i]) for i in range(len(S) - 1)]
    h_list = np.linspace(0, 2, num=11)
    TN = TensorNetwork.SixQubitTree(Entangler.IsingEntangler())
    fidelities = []
    for i, h in enumerate(h_list):
        ham = hamiltonians.ising_model(6, 1, h)
        multiplicity, E_exact, vs_exact = hamiltonians.exact_gs(ham)
        TN.params = params_data[i]
        v_opt = np.array(TN.make_state_vector())
        maxfid = 0
        for j in range(multiplicity):
            fid = abs( np.dot(v_opt.conj(), vs_exact[:, j]))**2
            if fid > maxfid:
                maxfid = fid
        fidelities.append(maxfid)

    plt.plot(h_list, fidelities)
    plt.xlabel('h')
    plt.ylabel('Fidelity')
    plt.title('VQE using Tree TN, six qubits Ising model \n GP_minimize block by block, 3 sweeps')
    plt.show()


def two_qubit_pauli(circ: QuantumCircuit, qreg: QuantumRegister, qubit_1: int, qubit_2: int,
                    pauli_1: int, pauli_2: int, angle: float) -> None:
    """
    Applies a two-qubit Pauli rotation to a given QuantumCircuit.
    Qiskit has ZZ rotation but nothing else. This is to fix that.

    :param circ: quantum circuit
    :param qubit_1: number of the first qubit
    :param qubit_2: number of the second qubit
    :param pauli_1: number of the first operation. '0' is identity, '1' is pauli X and so on.
    :param pauli_2: number of the second operation.
    :param angle:
    :return: None
    """
    if pauli_1 == 0 or pauli_2 == 0:
        raise NotImplementedError

    paulituple = (pauli_1, pauli_2)
    qubitpair = (qubit_1, qubit_2)
    for i in range(2):
        if paulituple[i] == 3:
            pass
        elif paulituple[i] == 1:
            circ.h(qreg[qubitpair[i]])
        elif paulituple[i] == 2:
            circ.sdg(qreg[qubitpair[i]])
            circ.h(qreg[qubitpair[i]])

    # Replace this with RZZ when they fix the printing
    # They have, but it doesn't work for me for some reason
    circ.cx(qreg[qubit_1], qreg[qubit_2])
    circ.u1(angle, qreg[qubit_2])
    circ.cx(qreg[qubit_1], qreg[qubit_2])

    for i in range(2):
        if paulituple[i] == 3:
            pass
        elif paulituple[i] == 1:
            circ.h(qreg[qubitpair[i]])
        elif paulituple[i] == 2:
            circ.h(qreg[qubitpair[i]])
            circ.s(qreg[qubitpair[i]])
    # print(circ)


if __name__ == '__main__':
    q = QuantumRegister(5)
    circc = QuantumCircuit(q)
    two_qubit_pauli(circc, q, 3, 4, 2, 2, 0.5)
    print(circc)
