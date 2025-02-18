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

import flax.linen as nn
import jax.numpy as jnp
import jax

import netket as nk


class LogAmplitudes:
    """Classical ansatz for the CQD framework. Given a model for the log amplitudes, it computes the amplitudes and their gradients."""

    def __init__(self, model: nn.Module):
        """
        Args:
            model: flax model for the log amplitudes."""
        self.model = model

    @partial(jax.jit, static_argnums=0)
    def __call__(self, theta: dict, z: jnp.ndarray) -> jnp.ndarray:
        """Returns the amplitudes for the quantum samples.
        Args:
            theta: parameters of the model.
            z: quantum samples.
        """
        return jnp.exp(self.model.apply(theta, z))

    @partial(jax.jit, static_argnums=0)
    def gradient(self, theta: dict, z: jnp.ndarray) -> jnp.ndarray:
        """Returns the gradient of the amplitudes with respect to the parameters.
        Output shape: (n_samples, n_parameters)
        Args:
            theta: parameters of the model.
            z: quantum samples.
        """
        jac = nk.jax.jacobian(self.model.apply, theta["params"], z, mode="complex")
        # reshape final axis (if the parameters are matrices)
        jac = jax.tree.map(lambda x: x.reshape((x.shape[0], 2, -1)), jac)
        # Sum real and imagin
        jac = jax.tree.map(lambda x: x[:, 0, :] + 1j * x[:, 1, :], jac)
        jac = jax.tree.map(lambda x: jnp.einsum("i,ij->ij", self(theta, z), x), jac)
        jac = jax.tree.flatten(jac)[0]
        return jnp.concatenate(jac, axis=-1)
