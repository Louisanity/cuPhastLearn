import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize, approx_fprime
import csv
from itertools import chain

import sys
sys.path.append("..")

import Entangler
import TensorNetwork
import hamiltonians
# import TNOptimize
plt.rcParams.update({'font.size': 14})

import pennylane as qml
import pennylane.numpy as np

n_qubits = 10
wires = list(range(n_qubits))
depth = 4

ent = Entangler.IsingEntangler()
TN = TensorNetwork.Checkerboard(wires, ent, depth=depth)

conv_tol = 1e-6
method = "L-BFGS-B"

cuda = False

start_time = time.time()

J = 1

# np.random.seed(0)

h_vals = np.linspace(0, 2, num=101)

h_iter = chain(h_vals, reversed(h_vals))

init_params = np.random.rand(TN.n_params)

datetime = time.strftime("%Y-%m-%d_%H-%M-%S")

h_base = hamiltonians.xxz_heisenberg_model(n_qubits, 1, 0)
H_base = hamiltonians.explicit_hamiltonian(h_base)

h_field = hamiltonians.xxz_heisenberg_model(n_qubits, 0, 1)
H_field = hamiltonians.explicit_hamiltonian(h_field)

with open("vqe_" + datetime + ".csv", "a", newline='') as fd:
    statewriter = csv.writer(fd, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i, h in enumerate(h_iter):
        if np.isclose(h, 1):
            continue
        H = H_base + h * H_field
        H_op  = qml.Hermitian(H, wires)
        
        if cuda == True:
            dev = qml.device("lightning.gpu",wires)
        else:
            dev = qml.device("default.qubit",wires)

        @qml.qnode(dev)
        def circuit(params, wires, state=False):
            TN.construct_circuit(params)
            if state:
                return qml.state()
            else:
                return qml.expval(H_op)

        def cost_fn(params):
            return circuit(params, wires)

        opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

        params = init_params

        for n in range(800):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            # Calculate difference between new and old energies
            conv = np.abs(energy - prev_energy)
            if conv <= conv_tol:
                break

        #state = circuit(params, wires, state=True)
        if (i < 10):
            print(i, n, conv, energy)

        if (h > 1):
            state_label = 1
        else:
            state_label = 0
        #total_data = np.concatenate((state, params, [h, energy, state_label]))
        total_data = np.concatenate((params, [h, energy, state_label]))
        statewriter.writerow(total_data)
        init_params = params

end_time = time.time()
print("Execution time:", end_time - start_time)
