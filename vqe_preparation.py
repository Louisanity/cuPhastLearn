"""
This script performs Variational Quantum Eigensolver (VQE) simulations using Qiskit, Scipy, and custom modules.
It optimizes a quantum circuit to minimize the energy of a Hamiltonian representing the XXZ Heisenberg model,
with an alternating magnetic field applied. Results are saved to a CSV file with timestamp.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from itertools import chain
import csv
import sys

# Ensure custom modules are found assuming they are in the parent directory
sys.path.append("..")

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute

# Import custom modules
import Entangler
import TensorNetwork
import hamiltonians
import TNOptimize
import utils

# Update matplotlib settings for improved plot aesthetics
plt.rcParams.update({'font.size': 14})

def main():
    n_qubits = 10
    depth = 4

    # Prepare quantum and classical registers, and the tensor network
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    ent = Entangler.IsingEntangler()
    TN = TensorNetwork.Checkerboard(q, c, ent, depth=depth)

    # Optimization parameters
    tol = 1e-6
    method = "L-BFGS-B"

    # Define magnetic field values
    h_vals = np.linspace(0, 2, num=1001)
    h_iter = chain(h_vals, reversed(h_vals))

    # Initial parameters for the quantum circuit
    x_0 = np.random.rand(TN.n_params)

    # Timestamp for file naming
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Hamiltonian setup
    h_base = hamiltonians.xxz_heisenberg_model(n_qubits, 1, 0)
    H_base = hamiltonians.explicit_hamiltonian(h_base)

    h_field = hamiltonians.xxz_heisenberg_model(n_qubits, 0, 1)
    H_field = hamiltonians.explicit_hamiltonian(h_field)

    # Open a file to save the results
    with open("vqe_" + datetime + ".csv", "a", newline='') as fd:
        statewriter = csv.writer(fd, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i, h in enumerate(h_iter):
            if np.isclose(h, 1):
                continue
            H = H_base + h * H_field
            f = TNOptimize.build_objective_function(TN, explicit_H=H)
            res = minimize(f, x_0, options={'maxiter': 300}, tol=tol, method=method)
            circ = TN.construct_circuit(res.x)
            state = utils.get_state(circ)
            if (i % 10 == 0):
                print(f"Iteration: {i}, Energy: {res.fun}, Iterations: {res.nit}, Status: {res.message}")
            state_label = 1 if (h > 1) else 0
            total_data = np.concatenate((state, res.x, [h, res.fun, state_label]))
            statewriter.writerow(total_data)
            x_0 = res.x

if __name__ == "__main__":
    main()
