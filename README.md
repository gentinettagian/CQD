# Classically correcting Trotterized quantum dynamics (CQD)

## Overview
This repository contains the code accompanying the article on hybrid quantum-classical simulation of quantum many-body dynamics. The study presents a novel approach to overcoming current limitations in quantum computing by leveraging both quantum and classical computational resources. Our method utilizes Trotterization to evolve an initial state on a quantum computer, focusing on the Hamiltonian terms that are difficult to simulate classically. A classical model then corrects the quantum simulation by incorporating the omitted terms. This hybrid approach enhances scalability and mitigates noise while avoiding variational parameter optimization in the quantum circuit.


## Repository Structure
- `cqd/` - Contains the core implementation 
    - `expectation/` - Pauli strings and sums in the CQD framework, computation of expectation values by sampling from the quantum circuit
    - `forces/` - Contains the code to compute the forces and the quantum geometric tensor using the CQD ansatz
    - `tdvp/` - TDVP classes for one and two subsystems to run the simulation
    - `models/` - CQD Ansatz for one and two subsystems as well as a range of classical ansatze to plug into the framework
    - `integrators/` - Additional implicit Runge-Kutta integrators that are compatible with this framework
    - `utils.py` - Small helper functions used throughout the codebase
- `examples/` - Example codes used to simulations in the paper


## Installation
To run the code, ensure you have Python installed along with the required dependencies. Install dependencies using:
```bash
pip install -r requirements.txt
```

## Citation
If you use this code in your research, please cite our work:
```
@misc{gentinetta2025correctingextendingtrotterizedquantum,
      title={Correcting and extending Trotterized quantum many-body dynamics}, 
      author={Gian Gentinetta and Friederike Metz and Giuseppe Carleo},
      year={2025},
      eprint={2502.13784},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2502.13784}, 
}
```

## License
This repository is licensed under the Apache License, Version 2.0. See `LICENSE` for details.

## Contact
For questions or contributions, please contact the authors or open an issue in the repository.

