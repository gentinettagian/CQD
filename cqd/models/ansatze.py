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

import flax.linen as nn
import jax.numpy as jnp

from netket.utils.types import Array, NNInitFunc
from jax.nn.initializers import zeros


class JastrowPlusSingle(nn.Module):
    r"""
    Jastrow wave function :math:`\Psi(s) = \exp(\sum_{i \neq j} s_i W_{ij} s_j)`,
    where W is a symmetric matrix.

    The matrix W is treated as low triangular to avoid
    redundant parameters in the computation.
    """

    """The dtype of the weights."""
    kernel_init: NNInitFunc = zeros
    """Initializer for the weights."""

    @nn.compact
    def __call__(self, x_in: Array):
        nv = x_in.shape[-1]
        il = jnp.tril_indices(nv, k=-1)

        kernel = self.param(
            "real", self.kernel_init, (nv * (nv - 1) // 2,), float
        ) + 1j * self.param("imag", self.kernel_init, (nv * (nv - 1) // 2,), float)
        one_body = self.param(
            "one_body_real", self.kernel_init, (nv,), float
        ) + 1j * self.param("one_body_imag", self.kernel_init, (nv,), float)

        W = jnp.zeros((nv, nv), dtype=jnp.complex128).at[il].set(kernel)
        y = jnp.einsum("...i,ij,...j", x_in, W, x_in) + jnp.dot(x_in, one_body)

        return y

    def __repr__(self):
        return "JastrowPlusSingle"


class ExactState(nn.Module):
    """A simple ansatz that returns the exact wave function."""

    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        hilb_size = 2**n_sites

        log_amplitudes = self.param(
            "real", nn.initializers.normal(), (hilb_size,), float
        ) + 1j * self.param("imag", nn.initializers.normal(), (hilb_size,), float)

        offsets = 2 ** jnp.arange(n_sites - 1, -1, -1, dtype=jnp.int32)

        idx = jnp.sum(offsets * jnp.asarray((1 - x) / 2, dtype=jnp.int32), axis=-1)

        return log_amplitudes[idx].T
