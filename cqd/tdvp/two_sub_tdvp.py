# Copyright 2025 Gian Gentinetta - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import netket as nk
import jax
from netket.experimental.dynamics import RKIntegratorConfig

from cqd.models import TwoSubCQD
from cqd.expectation import PauliSum
from cqd.integrators import ImpRKConfig
from cqd.utils import zero_tree_like
from cqd.forces import ExactTwoSubForcesAndQGT, SampleTwoSubForcesAndQGT


from .base_tdvp import BaseTDVP
from .trotter import TrotterTerm, get_trotter_decomp, n_th_order_trotter


class TwoSubTDVP(BaseTDVP):
    """
    Implements the TDVP for the CQD ansatz on two-subsystem case.
    """

    def __init__(
        self,
        model: TwoSubCQD,
        initial_parameters: dict,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        init_quantum_state: np.ndarray | None = None,
        shots: int | None = None,
        samples: int | None = None,
        acond: float = 1e-10,
        rcond: float = 1e-10,
        integrator: RKIntegratorConfig | ImpRKConfig | None = None,
        trotter_step: float | None = None,
        trotter_order=1,
        correct_trotter: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            model: The CQD model for the two-subsystem case.
            initial_parameters: Initial parameters of the classical model
            h_tot: The total Hamiltonian of the system
            h_tilde: The tilde Hamiltonian acting on the quantum system.
            init_quantum_state: Initial quantum state of the system
            shots: Number of shots for the expectation value estimation. Set to `None` for exact evaluation.
            samples: Number of classical samples taken per shot.
            acond, rcond: Regularization parameters for the pseudo-inverse.
            integrator: Integrator configuration, defaults to Euler
            trotter_step: The time step for the trotterization. Set to `None` for exact evolution.
            trotter_order: Order of the Trotter decomposition
            correct_trotter: Whether to correct the trotterization
            seed: Random seed for the sampling
        """
        self.model = model
        self.n_quantum = model.n_quantum
        self.n_classical = model.n_classical

        if init_quantum_state is None:
            init_quantum_state = np.ones(2**self.n_quantum, dtype=np.complex128)
            init_quantum_state /= np.linalg.norm(init_quantum_state)

        if trotter_step:
            decomposition = get_trotter_decomp(h_tilde)
            if len(decomposition) == 2:
                trotter_terms = n_th_order_trotter(
                    trotter_order,
                    decomposition[0].simplify(),
                    decomposition[1].simplify(),
                )
            else:
                if trotter_order != 1:
                    raise NotImplementedError(
                        "Higher order Trotter decomposition not implemented for Hamiltonians with more than 2 Trotter terms."
                    )
                trotter_terms = [
                    TrotterTerm(
                        name=f"Term_{i}",
                        time_fraction=1.0,
                        paulis=group.simplify(),
                    )
                    for i, group in enumerate(decomposition)
                ]
        else:
            trotter_terms = None

        super().__init__(
            initial_parameters=initial_parameters,
            init_quantum_state=init_quantum_state,
            integrator=integrator,
            acond=acond,
            rcond=rcond,
            h_tilde=h_tilde,
            trotter_step=trotter_step,
            trotter_terms=trotter_terms,
            correct_trotter=correct_trotter,
            shots=shots,
        )

        if shots is None:
            self._force_generator = ExactTwoSubForcesAndQGT(self.model, h_tot, h_tilde)

        else:
            if samples is None:
                raise ValueError("If shots is not None, samples must be provided.")
            self._force_generator = SampleTwoSubForcesAndQGT(
                self.model, h_tot, h_tilde, shots, samples, jax.random.PRNGKey(seed)
            )

    def _forces_and_qgt(self, t: float, theta: dict):
        if self.trotter_step and self.correct_trotter:
            return self._force_generator(self.phi_q, theta, self.h_tilde())
        return self._force_generator(self.phi_q, theta)
