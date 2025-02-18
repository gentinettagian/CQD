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

import pennylane as qml
from pennylane.ops import PauliX, PauliY, PauliZ, Prod
from pennylane.fermi import FermiSentence
import jax.numpy as jnp
import jax

import netket as nk

from cqd.utils import fill_pauli_with_identities, split_pauli
from .sparse_circuits import *


def get_star_qubit(pauli_str: Prod) -> int:
    """Get the index of the first qubit where the Pauli operator is non-diagonal.
    Returns -1 if the operator is diagonal.
    """
    for i, op in enumerate(pauli_str.operands):
        if isinstance(op, (PauliX, PauliY)):
            return op.wires[0]
    return -1


class PauliString:
    """Class representing a Pauli string in the CQD formalism."""

    def __init__(self, pauli_str: Prod, n_qubits: int):
        """
        Args:
            pauli_str: Pennylane Pauli string
            n_qubits: Number of qubits in the system
        """
        self.pennylane = fill_pauli_with_identities(
            pauli_str.simplify(), range(n_qubits)
        )
        self.n_qubits = n_qubits
        self.star_qubit = get_star_qubit(self.pennylane)
        # Measurement instructions are only calcualted when needed
        self._measurement_instructions = None
        self._i_measurement_instructions = None

    @partial(jax.jit, static_argnums=(0,))
    def z_tilde(self, z: jnp.ndarray) -> tuple[list[int], complex]:
        """
        Returns the bitstring z_tilde and the phase S(z), s.t
        Pz = S(z_tilde)z_tilde
        Args:
            z: Array of bitstrings, shape (n_samples, n_qubits)
        """
        if isinstance(self.pennylane, Prod):
            operands = self.pennylane.operands
        else:
            operands = [self.pennylane]
        x_filter = jnp.array([isinstance(op, PauliX) for op in operands])
        y_filter = jnp.array([isinstance(op, PauliY) for op in operands])
        z_filter = jnp.array([isinstance(op, PauliZ) for op in operands])

        # Flip bits where PauliX or PauliY is present
        z_tilde = jnp.where(x_filter + y_filter, -z, z)

        # Compute the phase
        phase = jnp.prod(jnp.where(z_filter, z, 1), axis=-1)
        phase *= jnp.prod(jnp.where(y_filter, 1j * z, 1), axis=-1)

        return z_tilde, jnp.conj(phase)

    @property
    def measurement_instructions(self):
        """Return the instructions to measure the real part of off-diagonal pauli strings"""
        if self._measurement_instructions is None:
            self._measurement_instructions = self._get_measurements_instructions()
        return self._measurement_instructions

    @property
    def i_measurement_instructions(self):
        """Return the instructions to measure the imaginary part of off-diagonal pauli strings"""
        if self._i_measurement_instructions is None:
            self._i_measurement_instructions = self._get_measurements_instructions(
                imaginary_part=True
            )
        return self._i_measurement_instructions

    def mel(self, bitstring_1: jnp.ndarray, bitstring_2: jnp.ndarray) -> complex:
        """Compute the matrix element <z_1|P|z_2>"""
        z_tilde, phase = self.z_tilde(bitstring_2)
        if np.all(z_tilde == bitstring_1):
            return phase
        return 0

    def mels(self, basis: jnp.ndarray) -> jnp.ndarray:
        """Compute the matrix elements <z_i|P|z> for all z in the basis"""
        z_tilde, phase = self.z_tilde(basis)
        z_tilde = z_tilde * jnp.conj(phase).reshape((-1, 1))
        mat = jnp.dot(basis, z_tilde.T)
        return jnp.where(
            jnp.isclose(jnp.abs(mat), basis.shape[-1]), mat / basis.shape[-1], 0
        )

    def _get_measurements_instructions(self, imaginary_part=False):
        """Return the instructions to measure off-diagonal pauli strings"""
        operands = self.pennylane.operands
        instructions = []
        if self.star_qubit < 0:
            if imaginary_part:
                return None
            return instructions
        # Measure some quantity
        for i, op in enumerate(operands):
            if op.wires[0] == self.star_qubit:
                continue
            if isinstance(op, PauliX):
                instructions.append(
                    controlled_operator(sps_X, self.star_qubit, i, self.n_qubits)
                )
            elif isinstance(op, PauliY):
                instructions.append(
                    controlled_operator(sps_Y, self.star_qubit, i, self.n_qubits)
                )
            elif isinstance(op, PauliZ):
                instructions.append(
                    controlled_operator(sps_Z, self.star_qubit, i, self.n_qubits)
                )

        if imaginary_part:
            if isinstance(operands[self.star_qubit], PauliX):
                instructions.append(
                    single_qubit_operator(
                        sps_Rx(-jnp.pi / 2), self.star_qubit, self.n_qubits
                    )
                )
            elif isinstance(operands[self.star_qubit], PauliY):
                instructions.append(
                    single_qubit_operator(sps_H, self.star_qubit, self.n_qubits)
                )
        else:
            if isinstance(operands[self.star_qubit], PauliX):
                instructions.append(
                    single_qubit_operator(sps_H, self.star_qubit, self.n_qubits)
                )
            elif isinstance(operands[self.star_qubit], PauliY):
                instructions.append(
                    single_qubit_operator(
                        sps_Rx(jnp.pi / 2), self.star_qubit, self.n_qubits
                    )
                )
        return instructions

    def is_identity(self):
        return all(isinstance(op, qml.Identity) for op in self.pennylane.operands)

    def to_sparse(self):
        if self.is_identity():
            return sps.identity(2**self.n_qubits)
        return self.pennylane.sparse_matrix(range(self.n_qubits))

    def to_dense(self):
        return self.to_sparse().toarray()

    def __eq__(self, another):
        return hasattr(another, "pennylane") and self.pennylane == another.pennylane

    def __hash__(self):
        return hash(self.pennylane)

    def __add__(self, another):
        if isinstance(another, PauliString):
            return PauliSum([self, another], [1.0, 1.0])
        else:
            return another + self

    def __mul__(self, another: complex):
        return PauliSum([self], [another])

    def __rmul__(self, another):
        return self * another

    def __repr__(self):
        representation = ""
        wire = 0
        for op in self.pennylane.operands:
            assert op.wires[0] == wire
            wire += 1
            representation += str(op)[0]

        return representation

    def __sub__(self, another):
        return self + another * -1

    def to_json(self):
        if self.is_identity():
            return "Identity"
        ops = self.pennylane.operands
        paulis = []
        for op in ops:
            if isinstance(op, qml.Identity):
                continue
            if isinstance(op, PauliX):
                paulis.append({"op": "X", "qubit": op.wires[0]})
            if isinstance(op, PauliY):
                paulis.append({"op": "Y", "qubit": op.wires[0]})
            if isinstance(op, PauliZ):
                paulis.append({"op": "Z", "qubit": op.wires[0]})
        return paulis

    def split(self, partition: list[int]):
        """Split the PauliString into two PauliStrings, where the first Pauli lies in the given partition"""
        pauli1, pauli2 = split_pauli(self.pennylane, partition)
        new_qubits = len(partition)
        return PauliString(pauli1, new_qubits), PauliString(
            pauli2, self.n_qubits - new_qubits
        )


def id_pauli(n_qubits: int) -> PauliString:
    """Return the identity Pauli string for n_qubits"""
    return PauliString(qml.Identity(n_qubits), n_qubits)


class PauliSum:
    """Class representing a sum of Pauli strings in the CQD formalism."""

    def __init__(
        self,
        paulis: list[PauliString] | None = None,
        weights: list[np.complex128] | None = None,
    ):
        """
        Args:
            paulis: List of Pauli strings
            weights: List of weights
        """
        if paulis is None:
            paulis = []
            self.n_qubits = None
        else:
            self.n_qubits = paulis[0].n_qubits
        if weights is None:
            weights = [1.0] * len(paulis)
        self.paulis = paulis

        self.weights = weights

    def __add__(self, another):
        new = self.copy()
        if self.n_qubits is None:
            new.n_qubits = another.n_qubits
        if another.n_qubits != new.n_qubits:
            raise ValueError("PauliSum objects must have the same number of qubits")
        if isinstance(another, PauliString):
            new.paulis.append(another)
            new.weights.append(1.0)
        elif isinstance(another, PauliSum):
            new.paulis.extend(another.paulis)
            new.weights.extend(another.weights)
        return new

    def __radd__(self, another):
        if another == 0:
            return self
        return self + another

    def __sub__(self, another):
        return self + another * -1

    def __mul__(self, another):
        new = self.copy()
        new.weights = [w * another for w in self.weights]
        return new

    def __rmul__(self, another):
        return self * another

    def to_sparse(self):
        return sum(w * p.to_sparse() for p, w in zip(self.paulis, self.weights))

    def to_dense(self):
        return self.to_sparse().toarray()

    def __repr__(self) -> str:
        return " + ".join(
            f"{w} * {p}" if w != 1 else f"{p}"
            for p, w in zip(self.paulis, self.weights)
        )

    def copy(self):
        return PauliSum(self.paulis.copy(), self.weights.copy())

    def simplify(self, tol=None) -> "PauliSum":
        """Simplify the PauliSum by combining terms with the same PauliString"""
        simplified = {}
        for p, w in zip(self.paulis, self.weights):
            if p in simplified:
                simplified[p] += w
            else:
                simplified[p] = w
            if simplified[p] == 0:
                del simplified[p]

        if tol is not None:
            simplified = {p: w for p, w in simplified.items() if np.abs(w) > tol}

        if len(simplified) == 0:
            return 0 * PauliSum(
                [PauliString(qml.Identity(self.n_qubits), self.n_qubits)], [0]
            )

        return PauliSum(list(simplified.keys()), list(simplified.values()))

    @staticmethod
    def from_pennylane(pennylane_sum: qml.ops.Sum):
        weights, paulis = pennylane_sum.terms()
        n_qubits = max(len(pennylane_sum.wires), max(pennylane_sum.wires) + 1)
        return PauliSum(
            [PauliString(pauli, n_qubits) for pauli in paulis], [w for w in weights]
        )

    @property
    def pennylane(self):
        res = 0
        for p, w in zip(self.paulis, self.weights):
            res += w * p.pennylane
        return res

    def to_json(self, tol=1e-10):
        """Convert the PauliSum to a JSON serializable format"""
        if not np.allclose(np.array(self.weights).imag, np.zeros(len(self.weights))):
            raise ValueError("Weights must be real")
        operators = []
        for p, w in zip(self.paulis, self.weights):
            if np.abs(w) < tol:
                continue
            operators.append({"pauli": p.to_json(), "weight": w.real})
        return {
            "operators": operators,
            "n_qubits": self.n_qubits,
        }

    def from_fermionic(fermi_sentence: FermiSentence, wire_map: dict[int, int]):
        """Transforms a fermionic operator into a Pauli sum using the Jordan-Wigner transformation"""
        fermi_words = []
        weights = []
        n_qubits = len(wire_map)

        for word, weight in fermi_sentence.items():
            fermi_words.append(word)
            weights.append(weight)
        paulisum = 0
        hopping_dict = {}
        for i, word in enumerate(fermi_words):
            if len(word) == 4:
                # Interaction term
                qubits = [key[1] for key in word.keys()]
                uniques = np.unique(qubits)
                if len(uniques) == 1:
                    continue
                if len(uniques) == 2:
                    pauli_word = (
                        PauliString(qml.I(wire_map[qubits[0]]), n_qubits)
                        - PauliString(qml.Z(wire_map[qubits[0]]), n_qubits)
                        - PauliString(qml.Z(wire_map[qubits[1]]), n_qubits)
                        + PauliString(
                            qml.Z(wire_map[qubits[0]]) @ qml.Z(wire_map[qubits[1]]),
                            n_qubits,
                        )
                    )
                    paulisum += weights[i] / 4 * pauli_word
            elif len(word) == 2:
                qubits = [key[1] for key in word.keys()]
                uniques = np.unique(qubits)
                if len(uniques) == 1:
                    # On-site term
                    pauli_word = PauliString(
                        qml.I(wire_map[qubits[0]]), n_qubits
                    ) - PauliString(qml.Z(wire_map[qubits[0]]), n_qubits)
                    paulisum += weights[i] / 2 * pauli_word
                else:
                    # Hopping term
                    min_qubit = min(qubits)
                    max_qubit = max(qubits)
                    if (min_qubit, max_qubit) not in hopping_dict:
                        hopping_dict[(min_qubit, max_qubit)] = 1
                        jw_string = qml.prod(
                            *[
                                qml.Z(wire_map[q])
                                for q in range(min_qubit + 1, max_qubit)
                            ]
                        )
                        x_str = (
                            qml.X(wire_map[min_qubit])
                            @ jw_string
                            @ qml.X(wire_map[max_qubit])
                        )
                        y_str = (
                            qml.Y(wire_map[min_qubit])
                            @ jw_string
                            @ qml.Y(wire_map[max_qubit])
                        )
                        paulisum += (
                            weights[i]
                            / 2
                            * (
                                PauliString(x_str, n_qubits)
                                + PauliString(y_str, n_qubits)
                            )
                        )
                    else:
                        # already processed
                        hopping_dict[(min_qubit, max_qubit)] += 1
            else:
                raise ValueError(f"Invalid fermionic word {word}")
        assert np.all(np.array(list(hopping_dict.values())) == 2)
        return paulisum

    def to_netket(self):
        """Converts the PauliSum to a NetKet Hamiltonian"""
        hilbert = nk.hilbert.Spin(s=0.5, N=self.n_qubits)
        opstrings = [str(p) for p in self.paulis]
        return nk.operator.PauliStringsJax(hilbert, opstrings, self.weights)
