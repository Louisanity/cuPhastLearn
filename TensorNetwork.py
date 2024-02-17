from qiskit import Aer, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import Entangler
from itertools import cycle, combinations_with_replacement, permutations, count
import numpy as np
# from utils import two_qubit_pauli
import utils

class TensorNetwork:
    """
    Tensor Network states prepared on qubits
    """
    backend = "qasm_simulator"
    
    def __init__(self, q, c):
        self.backend = "qasm_simulator"
        self.q = q
        self.c = c
        self.n_qubits = len(q)
    
    def set_params(self, params):
        if self.n_params == len(params):
            self.params = params
        else:
            raise ValueError('Incorrect number of parameters!')


class TNWithEntangler(TensorNetwork):
    '''Tensor network that requires an entangler'''
    def __init__(self, q, c, entangler):
        super().__init__(q, c)
        if not isinstance(entangler, Entangler.Entangler):
            raise TypeError('Should supply an entangler')
        self.entangler = entangler
        

class TestTN(TensorNetwork):

    def __init__(self, q, c, ent):
        super().__init__(q, c)
        self.entangler = ent
        self.params = [0]


    def construct_circuit(self, params=None):
        ps = params if params is not None else self.params
        circ = QuantumCircuit(self.q, self.c)
        circ.ry(ps[0], self.q[0])
        circ.cx(self.q[0], self.q[1])
        return circ


class Checkerboard(TNWithEntangler):
    '''
    Tensor network where gates are applied in a checkerboard pattern
    '''

    def __init__(self, q, c, entangler, params=None, depth=4):
        super().__init__(q, c, entangler)
        self.depth = depth
        self.n_tensors = (self.n_qubits // 2) * depth
        self.n_params = self.n_tensors * entangler.n_params
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

    def construct_circuit(self, params=None):
        ps = params if params is not None else self.params
        assert len(ps) == self.n_params, 'Incorrect number of parameters!'
        pc = cycle(ps)
        circ = QuantumCircuit(self.q, self.c)

        def bulknext(pc):
            return ([next(pc) for j in range(self.entangler.n_params)])

        for layer in range(self.depth):
            # remainder = layer % 2
            n_gates = (self.n_qubits) // 2
            for gate in range(n_gates):
                first_qubit = (gate * 2 + layer) % self.n_qubits
                second_qubit = (first_qubit + 1) % self.n_qubits
                self.entangler.apply(self.q, circ, first_qubit, second_qubit, bulknext(pc))
                circ.barrier()
            circ.barrier()
        return circ

    
class RankOne(TensorNetwork):
    '''Tensor network for building rank-1 (unentangled) states'''
    def __init__(self, q, c):
        super().__init__(q, c)
        self.n_params = len(q) * 2
        self.params = [0] * self.n_params
        self.n_tensors = self.n_qubits

    def construct_circuit(self, params=None):
        ps = params if params is not None else self.params
        assert len(ps) == self.n_params, 'Incorrect number of parameters!'
        pc = cycle(ps)
        circ = QuantumCircuit(self.q, self.c)
        for i in range(self.n_qubits):
            circ.ry(next(pc), self.q[i])
            circ.rz(next(pc), self.q[i])
        return circ


class TreeTN(TNWithEntangler):
    """Tree tensor network"""
    def __init__(self, q, c, entangler):
        super().__init__(q, c, entangler)
        self.full_layers = int(np.floor(np.log2(self.n_qubits)))
        self.n_tensors = self.n_qubits - 1
        self.n_params = self.n_tensors * entangler.n_params
        self.params = [0] * self.n_params

    def construct_circuit(self, params=None):
        ps = params if params is not None else self.params
        assert len(ps) == self.n_params, 'Incorrect number of parameters!'
        pc = cycle(ps)
        circ = QuantumCircuit(self.q, self.c)


        def bulknext(pc):
            return ([next(pc) for j in range(self.entangler.n_params)])

        m = self.n_qubits - 2 ** self.full_layers
        last_layer_bonds = []
        for i in range(m):
            pos = self.n_qubits - 2 * m + 2 * i
            last_layer_bonds.append((pos, pos + 1))

        last_layer_ins = [i for i in range(self.n_qubits - 2 * m)]
        for b in last_layer_bonds:
            last_layer_ins.append(b[0])

        for i in range(self.full_layers):
            for j in range(2 ** i):
                pos = j * 2 ** (self.full_layers - i)
                shift = 2 ** (self.full_layers - i - 1)
                self.entangler.apply(self.q, circ, last_layer_ins[pos], last_layer_ins[pos + shift], bulknext(pc))
        circ.barrier()

        for b in last_layer_bonds:
            self.entangler.apply(self.q, circ, b[0], b[1], bulknext(pc))

        return circ


class UCCSD(TensorNetwork):
    """Unitary coupled cluster ansatz. Not really a tensor network. Consider renaming the classes."""
    def __init__(self, q, c):
        super().__init__(q, c)
        self.n_params = 6 * self.n_qubits ** 2 - 3 * self.n_qubits
        self.params = [0] * self.n_params
        self.n_tensors = self.n_params

    def construct_circuit(self, params=None, trotter_steps=1, relaxed=False):
        """
        Constructs the quantum circuit for the UCCSD ansatz

        :param params: variational parameters
        :param trotter_steps: Number of terms in Suzuki--Trotter formula
        :param relaxed: if True, parameters can be set independently on each Trotter step
        :return: QuantumCircuit
        """
        ps = params if params is not None else self.params

        if trotter_steps > 1:
            raise NotImplementedError
        if relaxed:
            raise NotImplementedError

        pc = (p / trotter_steps for p in ps)
        #pc = count(0)

        circ = QuantumCircuit(self.q, self.c)

        for i in range(self.n_qubits):
            # single step
            circ.rx(next(pc), self.q[i])
            circ.ry(next(pc), self.q[i])
            circ.rz(next(pc), self.q[i])

        circ.barrier()

        for pauli_1, pauli_2 in combinations_with_replacement([1, 2, 3], 2):
            for qubit_1, qubit_2 in permutations(range(self.n_qubits), 2):
                utils.two_qubit_pauli(circ, self.q, qubit_1, qubit_2,
                                pauli_1, pauli_2, next(pc))
                circ.barrier()
            circ.barrier()

        return circ




########### OLD CLASSES BELOW THIS POINT


class NQubitTree(TensorNetwork):
    '''
    Tree tensor network for arbitrary amount of qubits
    '''
    def __init__(self, entangler, n_qubits, params=None):
        self.entangler = entangler
        self.n_qubits = n_qubits
        self.full_layers = int(np.floor(np.log(n_qubits) / np.log(2)))
        self.n_tensors = self.n_qubits - 1
        self.n_params = self.n_tensors * entangler.n_params
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

    def apply(self, q, circ, params=None):
        if params is not None:
            ps = params
        else:
            ps = self.params
        pc = cycle(ps)

        def bulknext(pc):
            return ([next(pc) for j in range(self.entangler.n_params)])

        m = self.n_qubits - 2**self.full_layers
        last_layer_bonds = []
        for i in range(m):
            pos = self.n_qubits - 2 * m + 2 * i
            last_layer_bonds.append((pos, pos + 1))
    
        last_layer_ins = [i for i in range(self.n_qubits - 2 * m)]
        for b in last_layer_bonds:
            last_layer_ins.append(b[0])
        
        for i in range(self.full_layers):
            for j in range(2**i):
                pos = j * 2**(self.full_layers - i)
                shift = 2**(self.full_layers - i - 1)
                self.entangler.apply(q, circ, last_layer_ins[pos], last_layer_ins[pos + shift], bulknext(pc))
        circ.barrier()
        
        for b in last_layer_bonds:
            self.entangler.apply(q, circ, b[0], b[1], bulknext(pc))
    

class SixQubitTree(TensorNetwork):
    '''
    general tree to be implemented later
    '''
    def __init__(self, entangler, params=None):
        #self.backend = "qasm_simulator"
        self.n_tensors = 5
        self.n_params = self.n_tensors * entangler.n_params
        self.entangler = entangler
        self.n_qubits = 6
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

    #def set_params(self, params):
    #    if self.n_params == len(params):
    #        self.params = params
    #    else:
    #        raise ValueError('Incorrect number of parameters!')

    def apply(self, q, circ, params=None):
        '''Apply the tensor network circuit'''
        if params is not None:
            ps = params
        else:
            ps = self.params
        k = self.entangler.n_params
        self.entangler.apply(q, circ, 2, 3, ps[:k])
        circ.barrier()
        self.entangler.apply(q, circ, 1, 2, ps[k:2 * k])
        self.entangler.apply(q, circ, 3, 4, ps[2*k:3*k])
        circ.barrier()
        self.entangler.apply(q, circ, 0, 1, ps[3*k:4*k])
        self.entangler.apply(q, circ, 4, 5, ps[4*k:])


class MERA(TensorNetwork):
    pass


if __name__ == '__main__':
    # q = QuantumRegister(3)
    # c = ClassicalRegister(1)
    # TN = UCCSD(q, c)
    # circ = TN.construct_circuit()
    #
    # print(circ)
    # print(TN.n_params)
    # print(circ.depth())
    pass

        

