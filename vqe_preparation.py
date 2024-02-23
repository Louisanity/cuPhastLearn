"""
Quantum Variational Eigensolver (VQE) Simulation with Tensor Networks and Entanglement.

This script performs a VQE simulation using a checkerboard tensor network as the ansatz
for simulating the ground state of the XXZ Heisenberg model under a magnetic field. It
utilizes Pennylane for quantum circuit simulation and optimization, and supports CUDA for GPU acceleration.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from itertools import chain
import pennylane as qml
import pennylane.numpy as qnp  # Ensure compatibility with PennyLane's NumPy wrapper
from scipy.optimize import minimize, approx_fprime

# Custom modules
import sys
sys.path.append("..")
import Entangler
import TensorNetwork
import hamiltonians

import argparse

conv_tol = 1e-6
method = "L-BFGS-B"

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run VQE Simulation with Customizable Parameters.')
    parser.add_argument('--n_qubits', type=int, default=10, help='Number of qubits (default: 10).')
    parser.add_argument('--depth', type=int, default=4, help='Circuit depth (default: 4).')
    parser.add_argument("--cuda", type=bool, default=False, help="Use CUDA-enabled devices")
    return parser.parse_args()



def initialize_simulation():
    """Initializes the simulation parameters and quantum device."""
    args = get_args()  # Parse command-line arguments
    n_qubits = args.n_qubits
    depth = args.depth
    cuda = args.cuda
    ent = Entangler.IsingEntangler()
    wires = list(range(n_qubits))
    TN = TensorNetwork.Checkerboard(wires, ent, depth=depth)
    if cuda:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
    return TN, dev


def define_circuit(TN, H_op, wires):
    """Defines the VQE circuit using Pennylane."""
    args = get_args()  # Parse command-line arguments
    n_qubits = args.n_qubits
    depth = args.depth
    cuda = args.cuda
    
    if cuda == True:
        dev = qml.device("lightning.gpu",wires)
    else:
        dev = qml.device("default.qubit",wires)
        
    @qml.qnode(dev)
    def circuit(params, state=False):
        args = get_args()  # Parse command-line arguments
        n_qubits = args.n_qubits
        depth = args.depth
        cuda = args.cuda
        wires = list(range(n_qubits))
        TN.construct_circuit(params)
        if state:
            return qml.state()
        else:
            return qml.expval(H_op)
    return circuit


def optimize_circuit(circuit, init_params, H_op):
    """Optimizes the circuit parameters."""
    args = get_args()  # Parse command-line arguments
    n_qubits = args.n_qubits
    depth = args.depth
    cuda = args.cuda
    wires = list(range(n_qubits))
    opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)
    params = init_params
    for n in range(800):
        params, prev_energy = opt.step_and_cost(lambda x: circuit(x), params)
        energy = circuit(params)
        conv = np.abs(energy - prev_energy)
        if conv <= conv_tol:
            break
            #state = circuit(params, wires, state=True)
        if (n < 10):
            print(i)
            print(n)
            print(conv)
            print(energy)
    return params, energy, n, conv



def main():
    """Main function to run the VQE simulation."""
    args = get_args()  # Parse command-line arguments
    n_qubits = args.n_qubits
    depth = args.depth
    cuda = args.cuda
    wires = list(range(n_qubits))

    if cuda == True:
        dev = qml.device("lightning.gpu",wires)
    else:
        dev = qml.device("default.qubit",wires)
    TN, device = initialize_simulation()
    h_vals = np.linspace(0, 2, num=101)
    h_iter = chain(h_vals, reversed(h_vals))
    init_params = np.random.rand(TN.n_params)
    datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    h_base = hamiltonians.xxz_heisenberg_model(n_qubits, 1, 0)
    H_base = hamiltonians.explicit_hamiltonian(h_base)
    h_field = hamiltonians.xxz_heisenberg_model(n_qubits, 0, 1)
    H_field = hamiltonians.explicit_hamiltonian(h_field)

    with open(f"vqe_{datetime}.csv", "a", newline='') as fd:
        statewriter = csv.writer(fd, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for i, h in enumerate(h_iter):
            if np.isclose(h, 1):
                continue
            H = H_base + h * H_field
            H_op = qml.Hermitian(H, wires)
            circuit = define_circuit(TN, H_op, wires)
            params, energy, n, conv = optimize_circuit(circuit, init_params, H_op)
            state_label = 1 if h > 1 else 0
            total_data = np.concatenate((params, [h, energy, state_label]))
            statewriter.writerow(total_data)
            init_params = params

if __name__ == "__main__":
    main()
