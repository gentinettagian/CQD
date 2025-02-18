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
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax


from cqd.utils import jitted_spins_from_int
from .pauli_string import PauliString, id_pauli


def sample_from_wf(key, n_qubits, wf, shots):
    """Sample from a quantum state"""
    indices = jnp.arange(2**n_qubits)
    probs = jnp.abs(wf) ** 2
    probs /= jnp.sum(probs)
    sampled_states = jax.random.choice(key, indices, shape=(shots,), p=probs)
    spi = jitted_spins_from_int(n_qubits)
    return spi(sampled_states)


def sample(key, state, shots, pauli_str, imag=False):
    """Sample the state given measurement instructions"""
    n_qubits = int(jnp.round(jnp.log2(state.shape[0])))
    if imag:
        for instr in pauli_str.i_measurement_instructions:
            state = instr.dot(state)
    else:
        for instr in pauli_str.measurement_instructions:
            state = instr.dot(state)

    return sample_from_wf(key, n_qubits, state, shots)


def sample_pauli_expectation(
    pauli_str: PauliString,
    state: np.array,
    shots: int,
    f: callable,
    key: jnp.ndarray | None = None,
) -> float:
    """Compute the expectation value of a Pauli operator by sampling the circuit.

    Args:
        pauli_str: The Pauli operator to measure.
        state: The quantum state to sample from.
        shots: The number of samples to take.
        f: The function to apply to the samples.
            Output shape given n_samples input is (n_samples, f_shape)

    Returns:
        The expectation value
            \sum_{z0=1,z} f(z)S(z)<psi|z><\tilde{z}|psi> + f(\tidle{z})S(\tilde{z})<psi|\tilde{z}><z|psi>

    [1] https://arxiv.org/abs/2106.05105
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if pauli_str.star_qubit < 0:
        samples = sample(key, state, shots, pauli_str)
        _, phases = pauli_str.z_tilde(samples)
        # Sum over first axis only in case f is vectorized
        result = jnp.sum(
            f(samples).reshape((shots, -1)) * phases.reshape((shots, -1)), axis=0
        )
        return result
    key1, key2 = jax.random.split(key)
    # For non-diagonal operators we need to sample special circuits
    real_samples = sample(key1, state, shots, pauli_str, imag=False)
    imag_samples = sample(key2, state, shots, pauli_str, imag=True)

    # Real part
    star_qubit_signs = real_samples[
        :, pauli_str.star_qubit
    ]  # Result of X / Y measurement
    real_samples = real_samples.at[:, pauli_str.star_qubit].set(1)  # Fix the star qubit
    zts, _ = pauli_str.z_tilde(real_samples)
    result = 0.5 * jnp.sum(
        star_qubit_signs.reshape((shots, -1))
        * (f(real_samples) + f(zts)).reshape((shots, -1)),
        axis=0,
    )

    # Imaginary part
    star_qubit_signs = imag_samples[
        :, pauli_str.star_qubit
    ]  # Result of (-Y) / X measurement
    imag_samples = imag_samples.at[:, pauli_str.star_qubit].set(1)  # Fix the star qubit
    zts, _ = pauli_str.z_tilde(imag_samples)
    result -= 0.5j * jnp.sum(
        star_qubit_signs.reshape((shots, -1))
        * (f(imag_samples) - f(zts)).reshape((shots, -1)),
        axis=0,
    )

    return result


@dataclass
class ExpectationTask:
    """Describing a task of an expectation value to be computed with the CQD formalism.
    A task is given by the expression
    .. math::
        \sum_{z,z'} \psi^q(t,z)^*\psi^q(t,z')\bra{z}\hat{P}\ket{z'}f(z)
    for a given Pauli operator P and a function f.

    """

    pauli_str: PauliString | None
    f: callable
    result: complex | None = None


class HybridExpectation:
    """
    Class to compute expectation values of Pauli operators using the CQD formalism.
    """

    def __init__(
        self,
        state: np.ndarray | None,
        shots: int,
        n_qubits: int,
        key: jnp.ndarray | None = None,
    ):
        """
        Args:
            state: The quantum state to sample from.
            shots: The number of samples to take.
            n_qubits: The number of qubits in the state.
            key: The random key to use for sampling.
        """
        self.state = state
        self.shots = shots
        self.expectation_tasks: list[ExpectationTask] = []
        self.n_qubits = n_qubits
        if key is None:
            key = jax.random.PRNGKey(42)
        self.key = key

    def add_task(self, task: ExpectationTask):
        """Add a task to the queue of tasks to compute."""
        self.expectation_tasks.append(task)

    def add_tasks(self, tasks: list[ExpectationTask]):
        """Add a list of tasks to the queue of tasks to compute."""
        self.expectation_tasks.extend(tasks)

    def compute(self, verbose=False):
        """Compute the expectation values of the tasks in the queue."""
        if self.state is None:
            raise ValueError("No state provided.")
        # Filter out tasks that are identities or None
        batches_dict = self._make_batches()
        for key, tasks in batches_dict.items():
            if verbose:
                print(f"Computing batch {key}, with {len(tasks)} tasks.")
            self._send_batch(tasks)
        self.expectation_tasks = []

    def _make_batches(self):
        """Make batches of tasks that can be computed together."""
        dict_batches = {"trivial": []}
        for task in self.expectation_tasks:
            if task.pauli_str is None or task.pauli_str.is_identity():
                task.pauli_str = id_pauli(self.n_qubits)
                dict_batches["trivial"].append(task)
                continue
            if task.pauli_str not in dict_batches:
                dict_batches[task.pauli_str] = [task]
            else:
                dict_batches[task.pauli_str].append(task)
        return dict_batches

    def _send_batch(self, tasks: list[ExpectationTask]):
        """Estimate a batch of tasks."""
        test_zq = jnp.ones((1, self.n_qubits))
        shapes = [task.f(test_zq).shape for task in tasks]
        key, subkey = jax.random.split(self.key)
        self.key = subkey

        def f_batch(z_q):
            return jnp.hstack(
                [
                    task.f(z_q).reshape((-1, jnp.prod(jnp.array([*shapes[i]]))))
                    for i, task in enumerate(tasks)
                ]
            )

        batch_result = (
            sample_pauli_expectation(
                tasks[0].pauli_str, self.state, self.shots, f_batch, key
            )
            / self.shots
        )
        already_read = 0
        for i, task in enumerate(tasks):
            shape_length = jnp.prod(jnp.array([*shapes[i]]))
            task.result = batch_result[already_read : already_read + shape_length]
            already_read += shape_length
