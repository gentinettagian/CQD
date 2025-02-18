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


from cqd.expectation import (
    ExpectationTask,
    HybridExpectation,
    PauliSum,
)
from cqd.models import LogAmplitudes
from cqd.utils import all_z


class BaseForcesAndQGT:
    """Base class to compute the forces and QGT with the CQD formalism."""

    def __init__(
        self,
        model: LogAmplitudes,
        n_qubits: int,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        verbose: bool = False,
    ):
        self.model = model
        self.n_qubits = n_qubits
        self.verbose = verbose
        self.h_tot = h_tot
        self.h_tilde = h_tilde

    def __call__(
        self, psi_q: jnp.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Returns: forces, time_forces, qgt"""
        return NotImplementedError


class SampleForcesAndQGT(BaseForcesAndQGT):
    """Class to compute the forces and QGT with the CQD formalism."""

    def __init__(
        self,
        model: LogAmplitudes,
        n_qubits: int,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        shots: int,
        key: jnp.ndarray | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            model: The classical model for which we want to calculate forces and qgt.
            n_qubits: The number of qubits in the quantum circuit.
            h_tot: The total Hamiltonian.
            h_tilde: The Hamiltonian applied on the quantum device.
            shots: The number of shots to use to estimate the expectation values.
            key: The key to use for the quantum device
            verbose: If true, prints the output of the estimator.
        """
        super().__init__(model, n_qubits, h_tot, h_tilde, verbose)

        self._estimator = HybridExpectation(None, shots, n_qubits, key)

        self._weights_tot = np.array(h_tot.weights)
        self._quantum_paulis_tot = h_tot.paulis

        self._weights_tilde = np.array(h_tilde.weights)
        self._quantum_paulis_tilde = h_tilde.paulis

    def __call__(
        self, psi_q: np.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Args:
            psi_q: The quantum state at which to compute the forces and QGT.
            thetas: The parameters of the classical ansatz.
            new_h_tilde: The Hamiltonian currently acting on the quantum device.
        """
        if new_h_tilde is None:
            paulis_tilde = self._quantum_paulis_tilde
            weights_tilde = self._weights_tilde
        else:
            paulis_tilde = new_h_tilde.paulis
            weights_tilde = np.array(new_h_tilde.weights)
        self._estimator.state = psi_q
        # Create the tasks to be sent to the estimator
        qgt_ij_task, qgt_i_task, norm_task = self._qgt_tasks(thetas)
        forces_tasks, energy_tasks = self._forces_tasks(thetas)
        time_forces_tasks, time_energy_tasks = self._time_forces_tasks(
            thetas, paulis_tilde
        )

        # Compute the expectation values
        self._estimator.add_tasks(
            forces_tasks
            + energy_tasks
            + time_forces_tasks
            + time_energy_tasks
            + [qgt_ij_task, qgt_i_task, norm_task]
        )
        self._estimator.compute(verbose=self.verbose)
        # Reconstruct the forces and QGT
        forces = self._reconstruct_forces(
            forces_tasks, energy_tasks, qgt_i_task, norm_task, self._weights_tot
        )
        time_forces = self._reconstruct_forces(
            time_forces_tasks,
            time_energy_tasks,
            qgt_i_task,
            norm_task,
            weights_tilde,
        )
        qgt = self._reconstruct_qgt(qgt_i_task, qgt_ij_task, norm_task)
        return forces - time_forces, qgt

    def _forces_tasks(self, thetas: dict):
        """Create the tasks to compute the forces."""

        tasks_forces = [
            ExpectationTask(
                pauli,
                lambda z, pauli=pauli: self._f_force(z, pauli, thetas),
            )
            for pauli in self._quantum_paulis_tot
        ]
        tasks_energy = [
            ExpectationTask(
                pauli,
                lambda z, pauli=pauli: self._f_energy(z, pauli, thetas),
            )
            for pauli in self._quantum_paulis_tot
        ]

        return tasks_forces, tasks_energy

    def _time_forces_tasks(self, thetas: dict, paulis_tilde):
        """Create the tasks to compute the forces."""

        tasks_forces = [
            ExpectationTask(pauli, lambda z: self._f_time_force(z, thetas))
            for pauli in paulis_tilde
        ]
        tasks_energy = [
            ExpectationTask(pauli, lambda z: self._f_time_energy(z, thetas))
            for pauli in paulis_tilde
        ]

        return tasks_forces, tasks_energy

    def _qgt_tasks(self, thetas: dict):
        """Create the tasks to compute the QGT."""

        return (
            ExpectationTask(None, lambda z: self._f_qgt_ij(z, thetas)),
            ExpectationTask(None, lambda z: self._f_qgt_i(z, thetas)),
            ExpectationTask(None, lambda z: self._f_norm(z, thetas)),
        )

    def _reconstruct_forces(
        self, forces_tasks, energy_tasks, qgt_i_task, norm_task, weights
    ):
        """Reconstruct the forces from the expectation values."""
        forces = (
            np.sum(
                [weight * task.result for weight, task in zip(weights, forces_tasks)],
                axis=0,
            )
            / norm_task.result
        )
        energy = (
            np.sum(
                [weight * task.result for weight, task in zip(weights, energy_tasks)],
                axis=0,
            )
            / norm_task.result
        )
        return forces - energy * qgt_i_task.result / norm_task.result

    def _reconstruct_qgt(self, qgt_i_task, qgt_ij_task, norm_task):
        """Reconstruct the QGT from the expectation values."""
        qgt_i = qgt_i_task.result / norm_task.result
        qgt_ij = qgt_ij_task.result / norm_task.result
        return qgt_ij.reshape(qgt_i.shape[0], qgt_i.shape[0]) - jnp.outer(
            qgt_i, jnp.conj(qgt_i)
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _f_force(self, z, pauli, thetas):
        z_tild = pauli.z_tilde(z)[0]
        grads = self.model.gradient(thetas, z)
        phi_c_tild = self.model(thetas, z_tild)
        return jnp.einsum("zi,z->zi", jnp.conj(grads), phi_c_tild)

    @partial(jax.jit, static_argnums=(0, 2))
    def _f_energy(self, z, pauli, thetas):
        z_tild = pauli.z_tilde(z)[0]
        phi_c = self.model(thetas, z)
        phi_c_tild = self.model(thetas, z_tild)
        return jnp.conj(phi_c) * phi_c_tild

    @partial(jax.jit, static_argnums=(0))
    def _f_time_force(self, z, thetas):
        grads = self.model.gradient(thetas, z)
        phi_c = self.model(thetas, z)
        return jnp.einsum("zi,z->zi", jnp.conj(grads), phi_c)

    @partial(jax.jit, static_argnums=(0))
    def _f_time_energy(self, z, thetas):
        phi_c = self.model(thetas, z)
        return jnp.abs(phi_c) ** 2

    @partial(jax.jit, static_argnums=(0))
    def _f_qgt_ij(self, z, thetas):
        grads = self.model.gradient(thetas, z)
        return jnp.einsum("zi,zj->zij", jnp.conj(grads), grads)

    @partial(jax.jit, static_argnums=(0))
    def _f_qgt_i(self, z, thetas):
        phi_c = self.model(thetas, z)
        grads = self.model.gradient(thetas, z)
        return jnp.einsum("zi,z->zi", jnp.conj(grads), phi_c)

    @partial(jax.jit, static_argnums=(0))
    def _f_norm(self, z, thetas):
        phi_c = self.model(thetas, z)
        return jnp.abs(phi_c) ** 2


class ExactForcesAndQGT(BaseForcesAndQGT):
    """Class to compute the forces and QGT with the CQD formalism without shots."""

    def __init__(
        self,
        model: LogAmplitudes,
        n_qubits: int,
        h_tot: PauliSum,
        h_tilde: PauliSum,
        verbose: bool = False,
    ):
        """
        Args:
            circuit: The quantum circuit to sample, should result instructions given a time t.
            model: The classical model for which we want to calculate forces and qgt.
            n_qubits: The number of qubits in the quantum circuit.
            h_tot: The total Hamiltonian.
            h_tilde: The Hamiltonian applied on the quantum device.
        """
        super().__init__(model, n_qubits, h_tot, h_tilde, verbose)
        self._all_z = all_z(n_qubits)
        self._htot_mat = h_tot.to_sparse()
        self._htilde_mat = h_tilde.to_sparse()

    def __call__(
        self, psi_q: np.ndarray, thetas: dict, new_h_tilde: PauliSum | None = None
    ):
        """Compute the forces and QGT at a given time for parameters theta of the classical ansatz.
        Args:
            psi_q: The quantum state at which to compute the forces and QGT.
            thetas: The parameters of the classical ansatz.
            new_h_tilde: The Hamiltonian currently acting on the quantum device.
        """
        if new_h_tilde is None:
            h_tilde_mat = self._htilde_mat
        else:
            h_tilde_mat = new_h_tilde.to_sparse()
        psi_c = self.model(thetas, self._all_z)
        grad_c = self.model.gradient(thetas, self._all_z)
        forces = self._forces(psi_q, psi_c, grad_c, h_tilde_mat)
        qgt = self._qgt(psi_q, psi_c, grad_c)
        return forces, qgt

    # @partial(jax.jit, static_argnums=(0,))
    def _qgt(self, psi_q, psi_c, grad_c):
        """return <dl psi| dk psi> - <dl psi| psi><dk psi| psi>"""
        psi = psi_q * psi_c
        norm = jnp.linalg.norm(psi) ** 2
        grad_psi = psi_q.reshape((-1, 1)) * grad_c
        qgt = jnp.einsum("ki,kj->ij", jnp.conj(grad_psi), grad_psi) / norm
        shift = jnp.einsum("ki, k-> i", jnp.conj(grad_psi), psi) / norm
        qgtshift = jnp.einsum("i,j->ij", shift, jnp.conj(shift))
        return qgt - qgtshift

    # @partial(jax.jit, static_argnums=(0,))
    def _forces(self, psi_q, psi_c, grad_c, htilde_mat):
        """returns <dk psi | H | psi> - <dk psi | psi><psi | H | psi>"""
        psi = psi_q * psi_c
        norm = jnp.linalg.norm(psi) ** 2
        grad_psi = psi_q.reshape((-1, 1)) * grad_c

        h_tot_psi = self._htot_mat.dot(psi)
        forces = jnp.einsum("ki, k -> i", jnp.conj(grad_psi), h_tot_psi) / norm
        energy = jnp.einsum("i, i ->", jnp.conj(psi), h_tot_psi) / norm

        h_tild_psi_q = htilde_mat.dot(psi_q)
        time_forces = (
            jnp.einsum("ki,k, k -> i", jnp.conj(grad_psi), psi_c, h_tild_psi_q) / norm
        )
        time_energy = jnp.einsum("i, i ->", jnp.conj(psi) * psi_c, h_tild_psi_q) / norm
        forces = forces - time_forces
        energy = energy - time_energy
        shift = jnp.einsum("ki, k -> i", jnp.conj(grad_psi), psi) / norm

        return forces - shift * energy
