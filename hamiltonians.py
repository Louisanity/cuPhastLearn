from numpy import eye, array, diag, zeros, kron, complex64
from numpy.linalg import eig, eigh
from functools import reduce

import pennylane as qml
import pennylane.numpy as np
#from scipy.sparse import csr_matrix

def ising_model(n_spins, J, hx):
    ham = {}
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = J
    line = 'X' + 'I' * (n_spins - 1)
    if hx != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = hx
    return ham

def heis_model(n_spins, J, hx):
    ham = {}
    for spin in ['X', 'Y', 'Z']:
        line = spin + spin + 'I' * (n_spins - 2)
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = J
    line = 'X' + 'I' * (n_spins - 1)
    if hx != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = hx
    return ham

def xxz_heisenberg_model(n_spins, J_x, J_z):
    ham = {}
    for spin in ['X', 'Y']:
        line = spin + spin + 'I' * (n_spins - 2)
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = J_x
    line = 'Z' + 'Z' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = J_z
    return ham


def local_fields_hamiltonian(n_qubits, local_fields):
    """Adds a local magnetic field term on each qubit. Needs 3 * n_qubits real values"""
    ham = {}
    for i in range(n_qubits):
        for j, s in enumerate(['X', 'Y', 'Z']):
            key = "I" * i + s + "I" * (n_qubits - i - 1)
            ham[key] = local_fields[j + i * 3]
    return ham


def two_d_heisenberg(n_x, n_y, J):
    """Heisenberg model on a patch of square lattice. 
    Periodic boundary conditions"""
    ham = {}
    for i in range(n_x * n_y):
        if ((i + 1) % n_y == 0):
            horizontal_pair = (i, (i - n_y + 1))
        else:
            horizontal_pair = (i, (i + 1))
        if (i + n_y >= n_x * n_y):
            vertical_pair = (i, (i + n_y) % (n_x * n_y))
        else:
            vertical_pair = (i, (i + n_y))
        for s in ["X", "Y", "Z"]:
            key_list = ["I"] * n_x * n_y
            key_list[horizontal_pair[0]] = s
            key_list[horizontal_pair[1]] = s
            key = reduce(lambda a, b: a + b, key_list)
            ham[key] = J

            key_list = ["I"] * n_x * n_y
            key_list[vertical_pair[0]] = s
            key_list[vertical_pair[1]] = s
            key = reduce(lambda a, b: a + b, key_list)
            ham[key] = J
    return ham

        
def two_d_heisenberg_with_local_fields(n_x=3, n_y=3, J=1, local_fields=None):
    ham_1 = two_d_heisenberg(n_x, n_y, J)
    ham_2 = local_fields_hamiltonian(n_x * n_y, local_fields)
    return {**ham_1, **ham_2}


def magnetic_field(n_spins, h, direction='X'):
    """
    External magnetic field acting on all spins
    :param n_spins: qty of spins
    :param direction: 'X', 'Y', or 'Z'
    :param h: field
    :return: dictionary with terms
    """

    ham = {}
    line = direction + 'I' * (n_spins - 1)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = h
    return ham


def XY_model(n_spins, gamma, g):
    ''' XY model with transverse field'''
    ham = {}
    line = 'X' + 'X' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = 0.5 * (1 + gamma)
        
    line = 'Y' + 'Y' + 'I' * (n_spins - 2)
    for i in range(n_spins):
        term = line[-i:] + line[:-i]
        ham[term] = 0.5 * (1 - gamma)
        
    line = 'Z' + 'I' * (n_spins - 1)
    if g != 0:
        for i in range(n_spins):
            term = line[-i:] + line[:-i]
            ham[term] = g
    return ham

### ---------
'''
def explicit_hamiltonian(ham_dict):
    n_qubits = len(list(ham_dict.keys())[0])
    I = eye(2)
    X = array([[0, 1], [1, 0]])
    Y = array([[0, -1j], [1j, 0]])
    Z = diag([1, -1])
    pauli={}
    pauli['I'] = I
    pauli['X'] = X
    pauli['Y'] = Y
    pauli['Z'] = Z
    H = zeros((2**n_qubits, 2**n_qubits), dtype='complex128')
    for term, energy in ham_dict.items():
        matrices=[]
        for sym in term:
            matrices.append(pauli[sym])
        total_mat = energy * reduce(kron, matrices)
        H +=total_mat
    return H
'''
def explicit_hamiltonian(ham_dict):
    coeffs = list(ham_dict.values())
    obs = [qml.pauli.string_to_pauli_word(i) for i in ham_dict.keys()]
    H = qml.matrix(qml.Hamiltonian(coeffs, obs))
    return H

def exact_gs(ham_dict):
    H = explicit_hamiltonian(ham_dict)
    #    print(H)
    try:
        w, v = eigh(H)
    except:
        w, v = eig(H)
    multiplicity = list(w).count(w[0])
    return (multiplicity, w[0], v[:, :multiplicity])
'''
def qiskit_dict(ham_dict):
    label_coeff_list = []
    for label, value in ham_dict.items():
        label_coeff_list.append({'label':label,
                                 'coeff':
                                 {'real': value.real, 'imag':value.imag}})
    return {'paulis': label_coeff_list}
'''