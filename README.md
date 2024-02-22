# cuPhastLearn: Fast Phase-Transition Prediction with Quantum Machine Learning with Multi-GPU Acceleration

## Overview
This project explores the application of quantum computing to study condensed matter models, focusing on identifying phases and phase transitions in materials. Utilizing a quantum approach, we investigate the Heisenberg spin system defined by specific Hamiltonian dynamics, where interactions are characterized by \(J_{xy} < 0\) and \(J_z < 0\). Our goal is to analyze how properties of the system change under various conditions and identify discontinuities indicating phase transitions.

## Objective
To develop a quantum computing technique capable of identifying phase transitions in Heisenberg spin systems. This includes studying the phase diagram and applying machine learning algorithms to label phases of the system based on the provided datasets.

## Features
- **Quantum Circuit Implementation**: Design and implement quantum circuits to simulate Heisenberg spin systems.
- **Phase Diagram Analysis**: Analyze changes in system properties to identify phase transitions.
- **Machine Learning Integration**: Employ machine learning techniques to classify phases of spin systems.
- **Interactive Visualization**: Tools for visualizing phase diagrams and transitions.

## Installation
Clone the repository and install the required Python packages.

```bash
git clone https://github.com/Louisanity/QuantumPhastLearning.git
cd QuantumPhastLearning
pip install -r requirements.txt
```

## Usage
Run the main script to start the analysis or phase identification process for different molecule.

```bash
python main.py --molecule simulated_molecule
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Utilization of datasets and resources from PennyLane and cuQUantum on Heisenberg models and phase transitions.

## Resources
- [PennyLane's QML Data](https://docs.pennylane.ai/en/stable/code/qml_data.html)
- [Ising Model - Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
- [Phase Transition - Wikipedia](https://en.wikipedia.org/wiki/Phase_transition)
- [PennyLane Tutorial on Ising Model](https://pennylane.ai/qml/demos/tutorial_isingmodel_PyTorch/)
- [PennyLane Datasets](https://pennylane.ai/datasets/qspin/transverse-field-ising-model)

## Contact
For more information, reach out to us at:
- **Project Link**: [https://github.com/YourGitHubRepo/QuantumPhaseIdentifier](https://github.com/Louisanity/QuantumPhastLearning)
- **Email**: xxx@imperail.ac.uk

