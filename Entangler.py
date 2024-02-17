from itertools import cycle
from math import pi
import abc

## I never use apply with params=None, do I really need it?  The inits
## can go to the garbage as well. I could also stop making classes and
## just pass a function to the TensorNetwork constructor


class Entangler(metaclass=abc.ABCMeta):
    """Two-qubit gate used as an element of a TN state
    """
    def __init__(self, params=None):
        self.n_params = 2 
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

    def set_params(self, params):
        """Set the parameters of the gate"""
        if len(params) == self.n_params:
            self.params = params
        else:
            raise ValueError('Incorrect amount of parameters!')

    @abc.abstractmethod
    def apply(self, q, circ, i, j, params=None):
        """
        Apply with internal parameters if None
        otherwise use those supplied"""

class OptimalEntangler(Entangler):
    """Entangler capable of implementing anything from SU(4), which uses
    minimum amount of parameters and minimum amount of CNOTs. See
    https://arxiv.org/abs/quant-ph/0308006v3 for details

    """
    def __init__(self, params=None):
        self.n_params = 15
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

            
    def apply(self, q, circ, i, j, params):
        if len(params) == self.n_params:
            pc = cycle(params)
        else:
            raise ValueError('Incorrect number of parameters!')
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])

        circ.cx(q[j], q[i])
        circ.rz(next(pc), q[i])
        circ.ry(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.ry(next(pc), q[j])
        circ.cx(q[j], q[i])
        
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])
    

class IsingEntangler(Entangler):
    """Entangler that uses the operators found in
    the Ising model
    <| (exp(X) \otimes exp(X)) exp(ZZ) (exp(Z) \otimes exp(Z))
    """
    def __init__(self, params=None):
        self.n_params = 5
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)
    
    def apply(self, q, circ, i, j, params=None):
        """
        q - QuantumRegister
        circ - QuantumCircuit
        i, j - qubits to act on
        """
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')
        circ.rx(next(pc), q[i])
        circ.rx(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.rz(next(pc), q[i])
        circ.rz(next(pc), q[j])


class ConservingEntangler(Entangler):
    """Two Z rotations followed by an entangling two-qubit gate U2 from
    arxiv:1805.04340. When acting on states mapped from fermion Fock
    states, conserves particle number"""
    def __init__(self, params=None):
        self.n_params = 3
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)


    def apply(self, q, circ, i, j, params=None):
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')

        circ.rz(next(pc), q[i])
        circ.rz(next(pc), q[j])
        circ.cx(q[j], q[i])
        
        circ.h(q[j])
        circ.crz(next(pc), q[i], q[j])
        circ.h(q[j])
        
        circ.cx(q[j], q[i])
        
class ConservingEntanglerOne(Entangler):
    """Two-qubit gate U_1 from arxiv:1805.04340. 
    It conserves particle number, that is, a bitstring with a certain
    number of '1'-s can only be mapped to bitstrings with 
    the same number of these. Circuit implementation is mine."""
    def __init__(self, params=None):
        self.n_params = 2
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)


    def apply(self, q, circ, i, j, params=None):
        if params is None:
            theta = self.params[0]
            phi = self.params[1]
        else:
            if len(params) != 2:
                raise ValueError('Incorrect number of parameters!')
            theta = params[0]
            phi = params[1]

        circ.cx(q[j], q[i])
        circ.cx(q[i], q[j])
        
        circ.cz(q[i], q[j])
        circ.crz(-phi, q[i], q[j])
        
        ### Newest version of Qiskit has c-Ry gate, but this one doesn't
        circ.sdg(q[j])
        circ.h(q[j])
        circ.crz(theta, q[i], q[j])
        circ.h(q[j])
        circ.s(q[j])
        
        circ.crz(phi, q[i], q[j])
        
        circ.cx(q[i], q[j])
        circ.cx(q[j], q[i])
    
    

class AntiIsingEntangler(Entangler):
    """Entangler that uses some gates NOT found in the Ising model. This
    may or may not help converge faster

    """
    def __init__(self, params=None):
        self.n_params = 5
        if params is None:
            self.params = [0] * self.n_params
        else:
            super().set_params(params)

    def apply(self, q, circ, i, j, params=None):
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')
        circ.ry(next(pc), q[i])
        circ.ry(next(pc), q[j])
        
        ### exp(-i a XX)
        circ.h(q[i])
        circ.h(q[j])
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.h(q[i])
        circ.h(q[j])
        
        circ.rz(next(pc), q[i])
        circ.rz(next(pc), q[j])       

class XYEntangler(Entangler):
    """Entangler that uses the operators from XY model
    (technically ZY model)"""

    def __init__(self, params=None):
        self.n_params = 6
        if params is None:
            self.params = [0] * self.n_params
        else:
            super().set_params(params)

    def apply(self, q, circ, i, j, params=None):
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')
        circ.rx(next(pc), q[i])
        circ.rx(next(pc), q[j])
        
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])

        circ.u1(pi/2, q[i])
        circ.u1(pi/2, q[j])
        circ.h(q[i])
        circ.h(q[j])
        
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])
        
        circ.h(q[i])
        circ.h(q[j])
        circ.u1(-pi/2, q[i])
        circ.u1(-pi/2, q[j])
        
        circ.rz(next(pc), q[i])
        circ.rz(next(pc), q[j])

class R2MPOEntangler(Entangler):
    """Rank-2 matrix product operator.

    """
    def __init__(self, params=None):
        self.n_params = 7
        if params is None:
            self.params = [0] * self.n_params
        else:
            self.set_params(params)

    def apply(self, q, circ, i, j, params=None):
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')

        circ.ry(next(pc), q[i])
        circ.cx(q[i], q[j])
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])

class CartanEntangler(Entangler):
    """Two-qubit gate capable of implementing any SU(4) gate"""
    n_params = 15
    
    def apply(self, q, circ, i, j, params):
        if len(params) == self.n_params:
            pc = cycle(params)
        else:
            raise ValueError('Incorrect number of parameters!')
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])

        ### exp(-i a XX)
        circ.h(q[i])
        circ.h(q[j])
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.h(q[i])
        circ.h(q[j])

        ### exp(-i a ZZ)
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])

        ### exp(-i a YY)
        circ.s(q[i])
        circ.s(q[j])
        circ.h(q[i])
        circ.h(q[j])
        circ.cx(q[i], q[j])
        circ.u1(next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.h(q[i])
        circ.h(q[j])
        circ.sdg(q[i])
        circ.sdg(q[j])
        
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])
    

class CNOTEntangler(Entangler):
    """Like CartanEntangler, but in the middle it just has a CNOT"""
    n_params = 12

    def apply(self, q, circ, i, j, params):
        if len(params) == self.n_params:
            pc = cycle(params)
        else:
            raise ValueError('Incorrect number of parameters!')
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])
        circ.cx(q[i], q[j])
        circ.u3(next(pc), next(pc), next(pc), q[i])
        circ.u3(next(pc), next(pc), next(pc), q[j])

        
class SixParamEntangler(Entangler):
    """some wacky idea of mine"""
    def __init__(self, params=None):
        self.n_params = 6
        if params is None:
            self.params = [0] * self.n_params
        else:
            super().set_params(params)

    def apply(self, q, circ, i, j, params=None):
        if params is None:
            pc = cycle(self.params)
        else:
            if len(params) == self.n_params:
                pc = cycle(params)
            else:
                raise ValueError('Incorrect number of parameters!')
        circ.ry(next(pc), q[i])
        circ.ry(next(pc), q[i])
        circ.rz(next(pc), q[i])
        circ.rz(next(pc), q[i])
        circ.cz(q[i], q[j])
        circ.rx(next(pc), q[i])
        circ.rx(next(pc), q[i])
            
    
        
class DumbEntangler(Entangler):
    """Only applies Controlled-Z for visualization"""
    def __init__(self):
        """Should actually have zero parameters
        but for now will do
        """
        self.n_params = 1
        self.params = [0]

    def set_params(self, params):
        pass
    
    def apply(self, q, circ, i, j, params=None):
        circ.cz(q[i], q[j])
