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

from __future__ import annotations

import jax.numpy as jnp
import jax
from netket.experimental.dynamics import RKIntegratorConfig

import flax.linen as nn


from cqd.expectation import PauliSum, HybridExpectation, ExpectationTask
from cqd.forces import ExactForcesAndQGT, SampleForcesAndQGT
from cqd.integrators import ImpRKConfig
from cqd.models import LogAmplitudes
from cqd.utils import all_z

from .base_tdvp import BaseTDVP
from .trotter import TrotterTerm, n_th_order_trotter, get_trotter_decomp


class HybridTDVP(BaseTDVP):
    """Class implementing TDVP for the CQD ansatz."""

    def __init__(
        self,
        h_tot: PauliSum,
        classical_model: LogAmplitudes,
        initial_parameters: dict,
        init_quantum_state: jnp.ndarray | None = None,
        integrator: RKIntegratorConfig | ImpRKConfig = None,
        dt: float = 0.01,
        acond: float = 1e-5,
        rcond: float = 1e-4,
        h_tilde: PauliSum | None = None,
        trotter_step: float | None = None,
        trotter_order: int = 1,
        correct_trotter: bool = True,
        shots: int | None = None,
        seed: int = 42,
    ):
        """
        Args:
            h_tot: The total Hamiltonian of the system
            classical_model: The classical correction to the quantum state
            initial_parameters: Initial parameters of the classical model
            init_quantum_state: Initial quantum state of the system
            integrator: Integrator configuration, defaults to Euler
            dt: Time step of the integrator
            acond, rcond: Paramters for the inversion of the qgt
            h_tilde: The Hamiltonian used for the evolution of the quantum circuit
            trotter_step: Size of a single Trotter step (`None` for exact evoulution)
            trotter_order: Order of the Trotter decomposition
            correct_trotter: Whether to correct the Trotter error
            shots: Number of shots for the quantum circuit (`None` for statevector simulator)
            seed: Seed for the random number generator
        """
        self.n_qubits = h_tot.n_qubits
        self.model = classical_model
        if h_tilde is None:
            # The quantum circuit is evolving with the full Hamiltonian
            h_tilde = h_tot.copy()

        if init_quantum_state is None:
            init_quantum_state = jnp.ones(2**self.n_qubits, dtype=jnp.complex128)
            init_quantum_state /= jnp.linalg.norm(init_quantum_state)

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
            dt=dt,
            acond=acond,
            rcond=rcond,
            h_tilde=h_tilde,
            trotter_step=trotter_step,
            trotter_terms=trotter_terms,
            correct_trotter=correct_trotter,
            shots=shots,
        )
        self.key = jax.random.PRNGKey(seed)

        if shots is None:
            self._force_generator = ExactForcesAndQGT(
                self.model, self.n_qubits, h_tot, h_tilde
            )
        else:
            self._force_generator = SampleForcesAndQGT(
                self.model,
                self.n_qubits,
                h_tot,
                h_tilde,
                shots,
                self.key,
            )

    def _forces_and_qgt(self, t: float, theta: dict):
        if self.trotter_step and self.correct_trotter:
            return self._force_generator(self.phi_q, theta, self.h_tilde())
        return self._force_generator(self.phi_q, theta)

    def expect(self, pauli: PauliSum, theta: dict):
        """Compute the expectation value of a Pauli operator for the ansatz at the current time.
        Args:
            pauli: The Pauli operator to compute the expectation value of, as a `PauliSum`.
            theta: The parameters of the classical model.
        """
        if self.shots is None:
            z = all_z(self.n_qubits)
            psi = self.phi_q * self.model(theta, z)
            psi = psi / jnp.linalg.norm(psi)
            return jnp.vdot(psi, pauli.to_sparse().dot(psi))
        else:
            self.key, key = jax.random.split(self.key)
            expector = HybridExpectation(self.phi_q, self.shots, self.n_qubits, key)
            tasks = []
            for p in pauli.paulis:

                def f(z, p=p):
                    zt, _ = p.z_tilde(z)
                    return jnp.conj(self.model(theta, z)) * self.model(theta, zt)

                tasks.append(ExpectationTask(p, f))
            norm_task = ExpectationTask(
                None, lambda z: jnp.abs(self.model(theta, z)) ** 2
            )
            expector.add_tasks(tasks + [norm_task])
            expector.compute(verbose=False)
            norm = norm_task.result
            res = 0
            for w, task in zip(pauli.weights, tasks):
                res += w * task.result
            return res / norm
