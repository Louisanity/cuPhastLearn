"""
VQE Data Preparation Script

This script performs Variational Quantum Eigensolver (VQE) data preparation by iterating over a range of Hamiltonian parameters, optimizing a quantum circuit's parameters, and saving the results to a CSV file.

Usage:
  Run this script from the command line, providing values for the number of qubits (n_qubits), the circuit depth (depth), and a flag indicating whether to use CUDA-enabled devices (cuda).

  Example:
    python vqe_data_preparation.py --n_qubits 10 --depth 4 --cuda False

Arguments:
  --n_qubits INT: Number of qubits in the quantum circuit.
  --depth INT: Depth of the quantum circuit.
  --cuda BOOL: Whether to use CUDA-enabled devices. Pass True to enable.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from itertools import chain
import sys
import uuid
import json

# Append parent directory to system path for importing custom modules
sys.path.append("..")

import Entangler
import TensorNetwork
import hamiltonians
import utils

# Set matplotlib font size for plots
plt.rcParams.update({'font.size': 14})

import pennylane as qml

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run VQE data preparation with customizable parameters.")
parser.add_argument("--n_qubits", type=int, help="Number of qubits", required=True)
parser.add_argument("--depth", type=int, help="Depth of the quantum circuit", required=True)
parser.add_argument("--cuda", type=bool, help="Use CUDA-enabled devices", default=False)
args = parser.parse_args()

# Configuration variables from arguments
n_qubits = args.n_qubits
wires = list(range(n_qubits))
depth = args.depth
cuda = args.cuda

# Initialize the entangler and tensor network
ent = Entangler.IsingEntangler()
TN = TensorNetwork.Checkerboard(wires, ent, depth=depth)

# Optimization parameters and Hamiltonian setup
conv_tol = 1e-6
datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
h_vals = np.linspace(0, 2, num=101)
h_iter = chain(h_vals, reversed(h_vals))
init_params = np.random.rand(TN.n_params)

# Hamiltonians for the system
h_base = hamiltonians.xxz_heisenberg_model(n_qubits, 1, 0)
H_base = hamiltonians.explicit_hamiltonian(h_base)
h_field = hamiltonians.xxz_heisenberg_model(n_qubits, 0, 1)
H_field = hamiltonians.explicit_hamiltonian(h_field)

# CSV file for saving VQE data
with open(f"vqe_{datetime}.csv", "a", newline='') as fd:
    statewriter = csv.writer(fd, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i, h in enumerate(h_iter):
        if np.isclose(h, 1):
            continue
        H = H_base + h * H_field
        H_op = qml.Hermitian(H, wires)
        
        # Choose device based on CUDA flag
        if cuda:
            dev = qml.device("lightning.gpu", wires=wires)
        else:
            dev = qml.device("default.qubit", wires=wires)
        
        @qml.qnode(dev)
        def circuit(params, state=False):
            TN.construct_circuit(params)
            if state:
                return qml.state()
            return qml.expval(H_op)
        
        def cost_fn(params):
            return circuit(params)
        
        opt = qml.AdamOptimizer(stepsize=0.02)
        params = init_params
        
        for n in range(800):
            params, prev_energy = opt.step_and_cost(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)
            if conv <= conv_tol:
                break
            
        state_label = 1 if h > 1 else 0
        total_data = np.concatenate((params, [h, energy, state_label]))
        statewriter.writerow(total_data)
        init_params = params