from skopt import gp_minimize, forest_minimize, gbrt_minimize
from scipy.optimize import minimize #, dual_annealing, shgo
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import TensorNetwork
from math import pi
import TNOptimize
import hamiltonians
from numpy import array, float64, ndarray
# from qiskit_qcgpu_provider import QCGPUProvider

# Provider = QCGPUProvider()
Provider = None


from typing import Callable, List

sv_backend = Aer.get_backend("statevector_simulator")
# gpu_backend = Provider.get_backend("statevector_simulator")
gpu_backend = sv_backend
# sv_backend = Aer.get_backend("statevector_simulator")

def measure_ham_2(circ, ham_dict=None, explicit_H=None, shots=1000, backend="cpu"):
    '''Measures the expected value of a Hamiltonian passed either as
    ham_dict (uses QASM backend) or as explH (uses statevector backend)
    '''
    if ham_dict is not None:
        raise NotImplementedError("Use statevector backend for now")
    elif explicit_H is not None:
        if backend=="gpu":
            print("GPU support is disabled for now")
            job = execute(circ, gpu_backend)
        elif backend=="cpu":
            job = execute(circ, sv_backend)
        else:
            raise ValueError("gpu or cpu")
        result = job.result()
        state = result.get_statevector(circ)
        state = array(state).reshape((len(state), 1))
        E = (state.T.conj() @ explicit_H @ state)[0,0].real
        return E
    else:
        raise TypeError('pass at least something!')
        

def build_objective_function(TN: TensorNetwork.TensorNetwork,
                             explicit_H: ndarray, backend="cpu") -> Callable[[List[float]], float]:
    '''
    Takes the tensor network, Hamiltonian and returns a function R^k -> R. 
    '''
    def f(x):
        circ = TN.construct_circuit(x)
        return float64(measure_ham_2(circ, explicit_H=explicit_H, backend=backend))
    return f

def globalVQE_2(TN, ham_dict, use_explicit_H=True, n_calls=100, initial_circuit=None, verbose=True):

    def total_circ(x):
        if initial_circuit:
            return initial_circuit + TN.construct_circuit(x)
        else:
            return TN.construct_circuit(x)
            
    if use_explicit_H:
        H = hamiltonians.explicit_hamiltonian(ham_dict)
        def objective(x):
            return measure_ham_2(total_circ(x), explicit_H=H)
    else:
        def objective(x):
            return measure_ham_2(total_circ(x), ham_dict=ham_dict)

    res = gp_minimize(objective, [(0, 2 * pi)] * TN.n_params, n_calls=n_calls, verbose=verbose, x0=[0]*TN.n_params)
    return res

def any_order_VQE_2(TN: TensorNetwork, params_order: list,
                    init_vals=None,
                    ham_dict=None, explicit_H=None,
                    initial_circuit=None,
                    n_calls=100, verbose=True):
    '''
    Performs VQE by optimizing the tensor network in the order
    supplied in params_order.
    '''
    if init_vals:
        vals = [u for u in init_vals]
    else:
        vals = [0] * TN.n_params
    
    for free_parameters in params_order:
        print('Optimizing parameters ', free_parameters, end=' ... ')
        f = restrained_objective_2(TN, free_parameters, vals,
                                   explicit_H=explicit_H, ham_dict=ham_dict,
                                   initial_circuit=initial_circuit)
        suggested_point = [vals[i] for i in free_parameters]

        ## Supposedly the unitary exp(i F \theta) yields a pi-periodic
        ## cost function if F is a Pauli operator or any such that F**2 = 1
        ## Which is sadly not always the case
        res = gp_minimize(f, [(0,  2 * pi)] * len(free_parameters), n_calls=n_calls,
                          x0=suggested_point, verbose=verbose)

        
        #print(res.x)
        for i, n in enumerate(free_parameters):
            vals[n] = res.x[i]
        print('E = {0:0.4f}'.format(res.fun))
        #print(['{0:0.6f}'.format(v) for v in vals])
        #print(measure_ham_2(TN.construct_circuit(vals), explicit_H=explicit_H))

    return res.fun, vals
        
    

def restrained_objective_2(TN, free_parameters, default_vals, ham_dict=None, explicit_H=None, initial_circuit=None):
    '''Makes an objective function good for minimization. Locks most
    parameters, while leaving those listed in free_parameters as free
    '''

    def f(x):
        assert(len(x) == len(free_parameters)), 'Free parameters qty mismatch!'
        params = [u for u in default_vals]  ## this is probably bad
        for i, n in enumerate(free_parameters):
            params[n] = x[i]
        #print(params)
        circ = TN.construct_circuit(params)
        if initial_circuit:
            circ = initial_circuit + circ
        return measure_ham_2(circ, ham_dict=ham_dict, explicit_H=explicit_H)
    return f



############## old methods below


def get_en(q, c, ham_string, TN, shots=1000, circuit=None, pre_applied_circ=None):
    '''
    Prepares the tensor network state and measures
    the expected value of ham_string
    ham_string denotes a tensor product of Pauli matrices
    Only allowed to consist of I, X, Y, Z
    '''
    qasm_backend = Aer.get_backend('qasm_simulator')

    if circuit is None:
        circ = QuantumCircuit(q, c)
    else:
        circ = circuit
        
    #    circ = QuantumCircuit(q, c)

    circ.data = []
    if pre_applied_circ is not None:
        circ.data = pre_applied_circ.data
    TN.apply(q, circ)
    
    for i in range(len(ham_string)):
        if ham_string[i] == 'X':
            circ.h(q[i])
        if ham_string[i] == 'Y':
            circ.s(q[i])
            circ.h(q[i])
        if ham_string[i] != 'I':
            circ.measure(q[i], c[i])
    job = execute(circ, qasm_backend, shots=shots)
    result = job.result()
    answer = result.get_counts()
    expected_en = 0
    for key in answer.keys():
        expected_en += answer[key] * (-1)**key.count('1') / shots
    return expected_en

def measure_ham(q, c, ham_dict, TN, shots=1000, circuit=None, explH=None, pre_applied_circ=None):
    '''
    params_dict contains entries like
    "XIXZ": 0.234
    '''
    if circuit is None:
        circ = QuantumCircuit(q, c)
    else:
        circ = circuit
    
    if TN.backend == "qasm_simulator":        
        E = 0
        for key, value in ham_dict.items():
            E += value * get_en(q, c, key, TN, shots=shots, circuit=circ, pre_applied_circ=pre_applied_circ)
    elif TN.backend == "statevector_simulator":
#       circ = QuantumCircuit(q)
        circ.data = []
        if pre_applied_circ is not None:
            circ.data = pre_applied_circ.data
        TN.apply(q, circ)
        sv_backend = Aer.get_backend("statevector_simulator")
        job = execute(circ, sv_backend)
        result = job.result()
        state = result.get_statevector(circ)
        state = array(state).reshape((2**TN.n_qubits, 1))
        if explH is not None:
            H = explH
        else:
            H = hamiltonians.explicit_hamiltonian(ham_dict)
        E = (state.T.conj() @ H @ state)[0,0].real
            
    return E

def restrained_objective(x, ham_dict, TN, block, q, c):
    '''
    Objective function when only one gate is unlocked
    Not a pure function unfortunately. Works by updating TN
    '''
    assert(block <= TN.n_tensors)
    block_size = TN.entangler.n_params
    TN.params[block * block_size: (block + 1) * block_size] = x
    return measure_ham(q, c, ham_dict, TN)

def arb_index_objective(x, param_numbers, ham_dict, TN, q, c, circuit=None, explH=None):
    '''Assign values to any subset of parameters and measure'''
    for i, num in enumerate(param_numbers):
        TN.params[num] = x[i]
    return measure_ham(q, c, ham_dict, TN, circuit=circuit, explH=explH)

def any_order_VQE(ham_dict, TN, tensor_order=None, n_calls=20, method="GP"):
    """
    Performs VQE by optimizing the tensor network in the order
    supplied in tensor_order. The n_sweeps parameter is redundant as
    one can simply send a duplicate of the tuple or call this function
    again

    :param ham_dict: Hamiltonian to minimize
    :param TN: TensorNetwork used as ansatz
    :param tensor_order: the order of tensors. Can be a tuple with
    integers or tuples of integers. Integer means 'optimize this
    tensor', tuple means 'optimize these tensors jointly'
    :param method: method of optimization. Can be "GP", "local"
    :returns: value of the function. The `res` object is of limited
    meaning, the parameters are stored in TN

    """

    H = hamiltonians.explicit_hamiltonian(ham_dict)
    if not isinstance(TN, TensorNetwork.TensorNetwork):
        raise TypeError('Not a TensorNetwork')
    
    q = QuantumRegister(TN.n_qubits)
    c = ClassicalRegister(TN.n_qubits)
    circ = QuantumCircuit(q, c)

    block_size = TN.entangler.n_params
    n_entanglers = TN.n_tensors

    if tensor_order is None:
        order = tuple(i for i in range(n_entanglers))
    else:
        order = tensor_order

    for v in order:
        if isinstance(v, int):
            param_numbers = [v * block_size + i for i in
                             range(block_size)]
        elif isinstance(v, tuple):
            param_numbers = []
            for j in v:
                param_numbers += [j * block_size + i for i in
                                  range(block_size)]
        else:
            raise TypeError('tensor_order should be a tuple or nested tuple')
        print('Optimizing params', param_numbers, end=' -> ')
        def f(x):
            return arb_index_objective([float64(xi) for xi in x], param_numbers, ham_dict,
                                       TN, q, c, circuit=circ, explH=H)
        x0 = [TN.params[i] for i in param_numbers]
        
        res = gbrt_minimize(f, [(0, 2*pi)] * len(x0), x0=x0,
                          n_calls=n_calls, verbose=False)
        #res = shgo(f, [(0, 4*pi)] * len(x0))
        
        for i, num in enumerate(param_numbers):
            TN.params[num] = res.x[i]
        print('E = {0:0.4f}'.format(res.fun))

    return res.fun



def globalVQE(ham_dict, TN, initial_guess=None, n_calls=100, pre_applied_circ=None):
    """Optimizes the TN by simultaneously changing all the parameters
    """
    q = QuantumRegister(TN.n_qubits)
    c = ClassicalRegister(TN.n_qubits) 
    def objective(x):
        TN.params = x
        return measure_ham(q, c, ham_dict, TN, pre_applied_circ=pre_applied_circ)

    res = gbrt_minimize(objective, [(0, 2 * pi)] * TN.n_params, n_calls=n_calls, verbose=False, n_random_starts=30)
    return res.fun

#def globalVQE_with_prerun(ham_dict, TN, initial_guess=None, n_calls=100):
