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

import itertools

import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.ops import Prod


def comm(A, B):
    return A @ B - B @ A


def to01(z):
    return (1 - z) // 2


def from01(z):
    return 1 - 2 * z


def spins_from_int(i, n):
    return 1 - ((i // 2 ** jnp.arange(n)) % 2)[::-1] * 2


def jitted_spins_from_int(n):
    sfi = jax.vmap(lambda i: spins_from_int(i, n))
    return jax.jit(sfi)


def reorder_paulis(pauli_str: Prod) -> Prod:
    if len(pauli_str.operands) > 2:
        return pauli_str
    if pauli_str.operands[0].wires[0] > pauli_str.operands[1].wires[0]:
        return pauli_str.operands[1] @ pauli_str.operands[0]
    return pauli_str


def fill_pauli_with_identities(pauli_str: Prod, wires: list) -> Prod:
    if isinstance(pauli_str, Prod):
        pauli_str = reorder_paulis(pauli_str)
        operands = pauli_str.operands
    else:
        operands = [pauli_str]
    pauli_wires = pauli_str.wires
    added_paulis = 0
    if 0 in pauli_wires:
        full_pauli = operands[0]
        added_paulis += 1
    else:
        full_pauli = qml.I(0)
    for i in range(1, len(wires)):
        if i not in pauli_wires or len(operands) <= added_paulis:
            full_pauli = full_pauli @ qml.I(i)
        else:
            full_pauli = full_pauli @ operands[added_paulis]
            added_paulis += 1
    return full_pauli


def split_pauli(pauli, quantum_sites):
    """Split a Pauli string into a quantum and classical part."""
    n_quantum = len(quantum_sites)
    if isinstance(pauli, qml.ops.Prod):
        q_pauli = qml.I()
        c_pauli = qml.I()
        for i, op in enumerate(pauli.operands):
            if op.wires[0] in quantum_sites:
                q_pauli = q_pauli @ op
            else:
                c_pauli = c_pauli @ op
        return q_pauli, c_pauli.map_wires({i: i - n_quantum for i in pauli.wires})
    elif len(pauli.wires) == 0:
        return qml.I(), qml.I()
    elif pauli.wires[0] in quantum_sites:
        return pauli, qml.I()
    else:
        return qml.I(), pauli.map_wires({i: i - n_quantum for i in pauli.wires})


def convert_netket_convention(samples):
    return -samples


def state(circuit, wires):
    dev = qml.device("default.qubit.jax", wires=wires)

    @qml.qnode(dev)
    def state():
        circuit()
        return qml.state()

    return state()


def zero_tree_like(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pytree)


def random_tree_like(pytree, key, scale=1.0):
    key_generator = (
        key for key in jax.random.split(key, len(jax.tree_util.tree_flatten(pytree)[0]))
    )
    return jax.tree_util.tree_map(
        lambda x: scale * jax.random.normal(next(key_generator), x.shape),
        pytree,
    )


def all_z(n):
    return jnp.array(list(itertools.product([-1, 1], repeat=n)))


def get_single_excitations(hf_state):
    hf_state = jnp.array(hf_state)
    n_spins = len(hf_state)
    states = [hf_state]
    for i in range(n_spins):
        for j in range(i, n_spins):
            if hf_state[i] != hf_state[j]:
                new_string = hf_state.copy()
                new_string = new_string.at[i].set(hf_state[j])
                new_string = new_string.at[j].set(hf_state[i])
                states.append(new_string)
    return jnp.array(states)


def get_double_excitations(hf_state):
    n_spins = len(hf_state)
    states = []
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            for k in range(j + 1, n_spins):
                for l in range(k + 1, n_spins):
                    if hf_state[i] != hf_state[j] and hf_state[k] != hf_state[l]:
                        print(1, i, j, k, l)
                        new_string = hf_state.copy()
                        new_string = new_string.at[i].set(hf_state[j])
                        new_string = new_string.at[j].set(hf_state[i])
                        new_string = new_string.at[k].set(hf_state[l])
                        new_string = new_string.at[l].set(hf_state[k])
                        states.append(new_string)
                    elif hf_state[i] != hf_state[k] and hf_state[j] != hf_state[l]:
                        print(2, i, j, k, l)
                        new_string = hf_state.copy()
                        new_string = new_string.at[i].set(hf_state[k])
                        new_string = new_string.at[k].set(hf_state[i])
                        new_string = new_string.at[j].set(hf_state[l])
                        new_string = new_string.at[l].set(hf_state[j])
                        states.append(new_string)
                    elif hf_state[i] != hf_state[l] and hf_state[j] != hf_state[k]:
                        print(3, i, j, k, l)
                        new_string = hf_state.copy()
                        new_string = new_string.at[i].set(hf_state[l])
                        new_string = new_string.at[l].set(hf_state[i])
                        new_string = new_string.at[j].set(hf_state[k])
                        new_string = new_string.at[k].set(hf_state[j])
                        states.append(new_string)
    return jnp.array(states)


def sumabs(x):
    return jnp.sum(jnp.abs(x))


def get_structure(parameters: dict):
    """Return the structure of the parameters."""
    theta, tree_struct = jax.tree.flatten(parameters)
    shapes = [x.shape for x in theta]
    theta = [x.flatten() for x in theta]
    split_points = [len(theta[0])]
    for x in theta[1:]:
        split_points.append(split_points[-1] + len(x))
    return shapes, split_points, tree_struct
