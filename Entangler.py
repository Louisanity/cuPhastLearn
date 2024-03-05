import pennylane as qml
import pennylane.numpy as np

from itertools import cycle
from math import pi
import abc
## I never use apply with params=None, do I really need it?  The inits
## can go to the garbage as well. I could also stop making classes and
## just pass a function to the TensorNetwork constructor


class Entangler(metaclass=abc.ABCMeta):
    """Two-qubit gate used as an element of a TN state
    """
    def __init__(self):
        self.n_params = None
        #self.set_params(params)
    '''
    def set_params(self, params):
        """Set the parameters of the gate"""
        if len(params) == self.n_params:
            self.params = params
        else:
            raise ValueError('Incorrect number of parameters!')
    '''
    '''
    @abc.abstractmethod
    def apply(self, q, circ, i, j, params=None):
        """
        Apply with internal parameters if None
        otherwise use those supplied"""
    '''
    
class IsingEntangler():
    """Entangler that uses the operators found in
    the Ising model
    <| (exp(X) \otimes exp(X)) exp(ZZ) (exp(Z) \otimes exp(Z))
    """
    def __init__(self):
        self.n_params = 5
        
    def apply(self, params, wires):
        if len(params) != 5:
            raise ValueError('need 5 parameters!')
        if len(wires) != 2:
            raise ValueError('only works on 2 qubits!')
        pc = cycle(params)
        qml.RX(next(pc), wires[0])
        qml.RX(next(pc), wires[1])
        qml.CNOT(wires)
        qml.PhaseShift(next(pc), wires[1])
        qml.CNOT(wires)
        qml.RZ(next(pc), wires[0])
        qml.RZ(next(pc), wires[1])
        

class OptimalEntangler(Entangler):
    """Entangler capable of implementing anything from SU(4), which uses
    minimum amount of parameters and minimum amount of CNOTs. See
    https://arxiv.org/abs/quant-ph/0308006v3 for details

    """
    def __init__(self, params=None):
        self.n_params = 15
        self.set_params(params)

            
    def apply(self, params, wires):
        if len(params) == self.n_params:
            pc = cycle(params)
        else:
            raise ValueError('Need 15 parameters!')
        if len(wires)!=2:
            raise ValueError('Entangler only apply on two qubits!')
        qml.Rot(next(pc), next(pc), next(pc), wires[0])
        qml.Rot(next(pc), next(pc), next(pc), wires[1])

        qml.CNOT(wires[::-1])
        qml.RZ(next(pc), wires[0])
        qml.RY(next(pc), wires[1])
        qml.CNOT(wires)
        qml.RY(next(pc), wires[1])
        qml.CNOT(wires[::-1])
        
        qml.Rot(next(pc), next(pc), next(pc), wires[0])
        qml.Rot(next(pc), next(pc), next(pc), wires[1])
    



class ConservingEntangler(Entangler):
    """Two Z rotations followed by an entangling two-qubit gate U2 from
    arxiv:1805.04340. When acting on states mapped from fermion Fock
    states, conserves particle number"""
    def __init__(self, params=None):
        self.n_params = 3
        self.set_params(params)


    def apply(self, params, wires):
        if len(params) == self.n_params:
            pc = cycle(params)
        else:
            raise ValueError('need 3 parameters!')

        qml.RZ(next(pc), wires[i])
        qml.RZ(next(pc), wires[j])
        qml.RZ(wires[::-1])
        
        qml.Hadamard(wires[j])
        qml.CRZ(next(pc), wires)
        qml.Hadamard(wires[j])
        
        qml.CNOT(wires[::-1])
        
