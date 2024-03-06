import Entangler
from itertools import cycle, combinations_with_replacement, permutations, count
import numpy as np
# from utils import two_qubit_pauli
import pennylane as qml
import pennylane.numpy as np

class TensorNetwork:
    """
    Tensor Network states prepared on qubits
    """
    backend = "qasm_simulator"
    
    def __init__(self, wires):
        self.backend = "qasm_simulator"
        self.wires = wires
        self.n_qubits = len(wires)
    
    def set_params(self, params):
        if self.n_params == len(params):
            self.params = params
        else:
            raise ValueError('Incorrect number of parameters!')

class TNWithEntangler(TensorNetwork):
    '''Tensor network that requires an entangler'''
    def __init__(self, wires, entangler):
        super().__init__(wires)
        #if not isinstance(entangler, Entangler.Entangler):
        #raise TypeError('Should supply an entangler')
        self.entangler = entangler
        

class Checkerboard(TNWithEntangler):
    '''
    Tensor network where gates are applied in a checkerboard pattern
    '''

    def __init__(self, wires, entangler, params=None, depth=4):
        super().__init__(wires, entangler)
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

        def bulknext(pc):
            return ([next(pc) for j in range(self.entangler.n_params)])

        for layer in range(self.depth):
            # remainder = layer % 2
            n_gates = (self.n_qubits) // 2
            for gate in range(n_gates):
                first_qubit = (gate * 2 + layer) % self.n_qubits
                second_qubit = (first_qubit + 1) % self.n_qubits
                self.entangler.apply(bulknext(pc),wires=[first_qubit,second_qubit] )
                qml.Barrier(self.wires)
            qml.Barrier(self.wires)
            


