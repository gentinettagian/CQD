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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk


from cqd.expectation import (
    ExpectationTask,
    HybridExpectation,
    PauliSum,
    PauliString,
)
from cqd.models import TwoSubCQD


class BaseTwoSubForcesAndQGT:
    """Base class to compute the forces and QGT with the CQD formalism on two subsystems."""

    def __init__(
        self,
        cqd: TwoSubCQD,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        verbose: bool = False,
    ):
        self.cqd = cqd
        self.verbose = verbose
        self.h_tot = h_tot
        self.h_tilde = h_tilde

    def __call__(
        self, psi_q: jnp.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Returns: forces, time_forces, qgt"""

        return NotImplementedError


class ExactTwoSubForcesAndQGT(BaseTwoSubForcesAndQGT):
    """Class to compute the forces and QGT with the CQD formalism without shots on two subsystems.."""

    def __init__(
        self,
        cqd: TwoSubCQD,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        verbose: bool = False,
    ):
        """
        Args:
            cqd: The cqd model for which we want to calculate forces and qgt.
            h_tot: The total Hamiltonian.
            h_tilde: The Hamiltonian applied on the quantum device.
            verbose: If true, prints the progress of the estimator.
        """
        super().__init__(cqd, h_tot, h_tilde, verbose)
        self.h_tot_sp = h_tot.to_sparse()
        self.h_tilde_sp = h_tilde.to_sparse()

    def __call__(
        self, psi_q: np.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Args:
            psi_q: The quantum state at time t.
            thetas: The parameters of the classical ansatz.
        """

        psi = self.cqd.full_state(thetas, psi_q)
        log_grads = self.cqd.full_gradient(thetas)
        grads = jnp.einsum("zi, z->zi", log_grads, psi)
        if new_h_tilde is not None:
            h_tilde_m = new_h_tilde.to_sparse()
        else:
            h_tilde_m = self.h_tilde_sp

        h_psi = self.h_tot_sp.dot(psi)
        norm = jnp.sum(jnp.abs(psi) ** 2)
        forces = jnp.einsum("zi, z->i", grads.conj(), h_psi) / norm

        forces_2 = jnp.einsum("zi, z->i", grads.conj(), psi) / norm
        energy = jnp.einsum("i, i", psi.conj(), h_psi) / norm
        forces = forces - forces_2 * energy

        h_tilde_psiq = h_tilde_m.dot(psi_q)
        h_tilde_psi = self.cqd.full_state(thetas, h_tilde_psiq)
        time_forces = jnp.einsum("zi, z->i", grads.conj(), h_tilde_psi) / norm

        energy_tilde = jnp.einsum("i, i", psi.conj(), h_tilde_psi) / norm
        time_forces = time_forces - forces_2 * energy_tilde

        qgt = jnp.einsum("zi, zj->ij", grads.conj(), grads) / norm
        qgt_shift = jnp.einsum("i, j->ij", forces_2, forces_2.conj())
        qgt = qgt - qgt_shift
        return forces - time_forces, qgt


class SampleTwoSubForcesAndQGT(BaseTwoSubForcesAndQGT):
    """Class to compute the forces and QGT with the CQD formalism without shots on two subsystems."""

    def __init__(
        self,
        cqd: TwoSubCQD,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        shots: int,
        samples_per_shot: int,
        key: jax.random.PRNGKey,
        verbose: bool = False,
    ):
        """
        Args:
            cqd: The cqd model for which we want to calculate forces and qgt.
            h_tot: The total Hamiltonian.
            h_tilde: The Hamiltonian applied on the quantum device.
            shots: The number of shots to take on the quantum device.
            samples_per_shot: The number of classical samples to take per shot.
            verbose: If true, prints the progress of the estimator.
        """
        super().__init__(cqd, h_tot, h_tilde, verbose)
        self.shots = shots
        self.samples = samples_per_shot
        self.key, key = jax.random.split(key)
        self.expector = HybridExpectation(None, shots, h_tilde.n_qubits, key)

    def __call__(
        self, psi_q: np.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Args:
            psi_q: The quantum state at time t.
            thetas: The parameters of the classical ansatz.
        """
        if new_h_tilde is not None:
            h_tilde = new_h_tilde
        else:
            h_tilde = self.h_tilde

        forces_tasks, energy_tasks = self._forces_tasks(self.h_tot, thetas)
        time_forces_tasks, time_energy_tasks = self._time_forces_tasks(thetas, h_tilde)
        key1, key2, self.key = jax.random.split(self.key, 3)
        qgt_task = ExpectationTask(
            None, lambda z_q: _f_qgt(self.cqd, z_q, thetas, self.samples, key1)
        )
        qgt_shift_task = ExpectationTask(
            None, lambda z_q: _f_time(self.cqd, z_q, thetas, self.samples, key2)
        )
        norm_task = ExpectationTask(None, lambda z_q: _f_norm(self.cqd, z_q, thetas))

        # Send to backend and compute
        self.expector.state = psi_q
        self.expector.add_tasks(
            forces_tasks
            + energy_tasks
            + time_forces_tasks
            + time_energy_tasks
            + [qgt_task, qgt_shift_task, norm_task]
        )
        self.expector.compute(self.verbose)
        # Recombine results
        norm = norm_task.result
        forces = 0
        for weight, f_task, e_task in zip(
            self.h_tot.weights, forces_tasks, energy_tasks
        ):
            forces += weight * (
                f_task.result / norm - qgt_shift_task.result * e_task.result / norm**2
            )
        for weight, f_task, e_task in zip(
            h_tilde.weights, time_forces_tasks, time_energy_tasks
        ):
            forces -= weight * (
                f_task.result / norm - qgt_shift_task.result * e_task.result / norm**2
            )
        qgt = (
            qgt_task.result.reshape((forces.shape[0], forces.shape[0])) / norm
            - jnp.outer(qgt_shift_task.result, qgt_shift_task.result.conj()) / norm**2
        )

        return forces, qgt

    def _forces_tasks(self, h_tot: PauliSum, theta: dict) -> list[ExpectationTask]:

        forces_tasks = []
        energy_tasks = []

        for pauli in h_tot.paulis:
            pauli_q, pauli_c = pauli.split(list(range(self.cqd.n_quantum)))
            self.key, key = jax.random.split(self.key)
            f_f = lambda z_q, pauli_q=pauli_q, pauli_c=pauli_c, key=key: _f_pauli(
                self.cqd, pauli_q, pauli_c, z_q, theta, self.samples, key
            )
            self.key, key = jax.random.split(self.key)
            f_e = lambda z_q, pauli_q=pauli_q, pauli_c=pauli_c, key=key: _f_energy(
                self.cqd, pauli_q, pauli_c, z_q, theta, self.samples, key
            )
            forces_tasks.append(ExpectationTask(pauli_q, f_f))
            energy_tasks.append(ExpectationTask(pauli_q, f_e))

        return forces_tasks, energy_tasks

    def _time_forces_tasks(
        self, theta: dict, h_tilde: PauliSum
    ) -> list[ExpectationTask]:
        forces_tasks = []
        energy_tasks = []
        for pauli in h_tilde.paulis:
            self.key, key = jax.random.split(self.key)
            f = lambda z_q, key=key: _f_time(self.cqd, z_q, theta, self.samples, key)
            forces_tasks.append(ExpectationTask(pauli, f))
            energy_tasks.append(
                ExpectationTask(pauli, lambda z_q: _f_norm(self.cqd, z_q, theta))
            )

        return forces_tasks, energy_tasks


def _f_pauli(
    cqd: TwoSubCQD, pauli_q: PauliString, pauli_c: PauliString, z_q, theta, samples, key
):
    z_q, z_c = cqd.sample_z_c(theta, z_q, key, samples)
    q_amps = cqd.q_amp(theta, z_q)
    z_qt, _ = pauli_q.z_tilde(z_q)
    z_ct, phase = pauli_c.z_tilde(z_c)
    quotient = jnp.exp(
        cqd.model.apply(theta, z_qt, z_ct) - cqd.model.apply(theta, z_q, z_c)
    )
    grad = cqd.gradient(theta, z_q, z_c)
    return (
        jnp.einsum(
            "zs,zsi, zs, zs->zi",
            q_amps,
            grad.conj(),
            quotient,
            phase,
        )
        / samples
    )


def _f_energy(
    cqd: TwoSubCQD, pauli_q: PauliString, pauli_c: PauliString, z_q, theta, samples, key
):
    z_q, z_c = cqd.sample_z_c(theta, z_q, key, samples)
    q_amps = cqd.q_amp(theta, z_q)
    z_qt, _ = pauli_q.z_tilde(z_q)
    z_ct, phase = pauli_c.z_tilde(z_c)
    quotient = jnp.exp(
        cqd.model.apply(theta, z_qt, z_ct) - cqd.model.apply(theta, z_q, z_c)
    )
    return jnp.einsum("zs,zs, zs->z", q_amps, quotient, phase) / samples


def _f_time(cqd: TwoSubCQD, z_q, theta, samples, key):
    z_q, z_c = cqd.sample_z_c(theta, z_q, key, samples)
    q_amps = cqd.q_amp(theta, z_q)
    log_grads = cqd.gradient(theta, z_q, z_c)
    return jnp.einsum("zs, zsi->zi", q_amps, log_grads.conj()) / samples


def _f_norm(cqd: TwoSubCQD, z_q, theta):
    q_amps = cqd.q_amp(theta, z_q)
    return q_amps


def _f_qgt(cqd: TwoSubCQD, z_q, theta, samples, key):
    z_q, z_c = cqd.sample_z_c(theta, z_q, key, samples)
    q_amps = cqd.q_amp(theta, z_q)
    log_grads = cqd.gradient(theta, z_q, z_c)
    return (
        jnp.einsum("zs, zsi, zsj->zij", q_amps, log_grads.conj(), log_grads) / samples
    )
