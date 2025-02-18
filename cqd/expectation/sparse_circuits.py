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
import scipy.sparse as sps


def single_qubit_operator(gate, i, n):
    """Implement a single qubit gate as a sparse matrix"""
    left = sps.identity(2**i)
    right = sps.identity(2 ** (n - i - 1))
    mat = sps.kron(sps.kron(left, gate), right)

    # Explicitly convert to CSR since kron likes to return COO
    mat = sps.csr_array(mat)
    return mat


def controlled_operator(gate, control, target, n):
    """Implement a controlled gate as a sparse matrix"""
    if control < target:
        left = sps.identity(2**control)
        zero_proj = sps.csr_array(np.array([[1, 0], [0, 0]]))
        one_proj = sps.csr_array(np.array([[0, 0], [0, 1]]))
        middle = sps.identity(2 ** (target - control - 1))
        right = sps.identity(2 ** (n - target - 1))
        id_gate = sps.identity(2)
        cgate = sps.kron(sps.kron(zero_proj, middle), id_gate) + sps.kron(
            sps.kron(one_proj, middle), gate
        )
        mat = sps.kron(sps.kron(left, cgate), right)
    else:
        left = sps.identity(2**target)
        zero_proj = sps.csr_array(np.array([[1, 0], [0, 0]]))
        one_proj = sps.csr_array(np.array([[0, 0], [0, 1]]))
        middle = sps.identity(2 ** (control - target - 1))
        right = sps.identity(2 ** (n - control - 1))
        id_gate = sps.identity(2)
        cgate = sps.kron(sps.kron(id_gate, middle), zero_proj) + sps.kron(
            sps.kron(gate, middle), one_proj
        )
        mat = sps.kron(sps.kron(left, cgate), right)

    # Explicitly convert to CSR since kron likes to return COO
    mat = sps.csr_array(mat)
    return mat


sps_X = sps.csr_array([[0, 1], [1, 0]])
sps_Y = sps.csr_array([[0, -1j], [1j, 0]])
sps_Z = sps.csr_array([[1, 0], [0, -1]])
sps_H = 1 / np.sqrt(2) * sps.csr_array([[1, 1], [1, -1]])

sps_Rx = (
    lambda theta: np.cos(theta / 2) * sps.identity(2) - 1j * np.sin(theta / 2) * sps_X
)
sps_Ry = (
    lambda theta: np.cos(theta / 2) * sps.identity(2) - 1j * np.sin(theta / 2) * sps_Y
)
sps_Rz = (
    lambda theta: np.cos(theta / 2) * sps.identity(2) - 1j * np.sin(theta / 2) * sps_Z
)
