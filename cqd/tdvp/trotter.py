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

from dataclasses import dataclass

import scipy.sparse as sps
import numpy as np

from cqd.expectation import PauliSum
from cqd.utils import comm


@dataclass
class TrotterTerm:
    """Dataclass for a Trotter term in the Hamiltonian."""

    name: str
    """Name of the Trotter term (for debugging purposes)."""
    time_fraction: float
    """Fraction of the total time the term is active."""
    paulis: PauliSum
    """PauliSum object representing the Hamiltonian term."""

    def evolve_instructions(self, t: float):
        """Get the instructions for the time evolution unitary of the Trotter term."""
        instructions = []
        for weight, pauli in zip(self.paulis.weights, self.paulis.paulis):
            instr = (
                np.cos(t * weight) * sps.identity(2**pauli.n_qubits)
                - 1j * np.sin(t * weight) * pauli.to_sparse()
            )
            instructions.append(sps.csr_array(instr))
        return instructions

    def __hash__(self):
        return hash(self.name)


def get_trotter_decomp(ham: PauliSum) -> list[PauliSum]:
    """Decompose a Hamiltonian into groups of mutually commuting terms."""
    groups = []
    included_paulis = []
    for i, pauli in enumerate(ham.paulis):
        if i in included_paulis:
            continue
        group = ham.weights[i] * pauli
        included_paulis.append(i)
        for j, paulj in enumerate(ham.paulis):
            if j in included_paulis:
                continue
            commutator = comm(group.pennylane, paulj.pennylane)
            weights, _ = commutator.terms()
            if np.all(np.array(weights) == 0):
                group += ham.weights[j] * paulj
                included_paulis.append(j)
        groups.append(group)
    return groups


def rescale_trotter_terms(terms: list[TrotterTerm], s):
    """Rescale the time fractions of a list of Trotter terms to match the timestep `s`."""
    rescaled_terms = []
    sign = 1 if s > 0 else -1
    for term in terms:
        name = term.name if sign > 0 else f"-{term.name}"
        rescaled_terms.append(
            TrotterTerm(
                name,
                np.abs(s) * term.time_fraction,
                sign * term.paulis,
            )
        )
    return rescaled_terms


def n_th_order_trotter(
    n, H1: PauliSum, H2: PauliSum, name_1="H1", name_2="H2"
) -> list[TrotterTerm]:
    """Get the n-th order Trotter decomposition of a Hamiltonian for two non-commuting observables $H_1$ and $H_2$."""
    if n == 1:
        return [TrotterTerm(name_1, 1.0, H1), TrotterTerm(name_2, 1.0, H2)]
    if n == 2:
        return [
            TrotterTerm(name_1, 0.5, H1),
            TrotterTerm(name_2, 1.0, H2),
            TrotterTerm(name_1, 0.5, H1),
        ]
    if n % 2 != 0:
        raise ValueError("Only even orders or 1 are supported.")

    s = 1 / (4 - 4 ** (1 / (n - 1)))
    n_min_2_terms = n_th_order_trotter(n - 2, H1, H2, name_1, name_2)
    terms = []
    terms += rescale_trotter_terms(n_min_2_terms, s)
    terms += rescale_trotter_terms(n_min_2_terms, s)
    terms += rescale_trotter_terms(n_min_2_terms, 1 - 4 * s)
    terms += rescale_trotter_terms(n_min_2_terms, s)
    terms += rescale_trotter_terms(n_min_2_terms, s)
    return terms
