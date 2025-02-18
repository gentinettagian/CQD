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
import numpy as np
import jax

import netket as nk
from copy import copy
import pennylane as qml

from netket.utils.struct import field
from cqd.utils import all_z


class HybridMeanField(nn.Module):
    """
    Mean field model for the classical degrees of freedom.
    The parameters depend on the quantum degrees of freedom through
    a dense layer.
    """

    @nn.compact
    def __call__(self, z_q, z_c):
        """
        Compute the log amplitudes for
        Args:
            z_q: quantum degrees of freedom
            z_c: classical degrees of freedom
        """
        lam = nn.Dense(features=z_c.shape[-1], name="real")(z_q)
        p = nn.log_sigmoid(lam * z_c)
        phase = 1j * nn.Dense(features=z_c.shape[-1], name="imag")(z_q) * z_c
        return 0.5 * jnp.sum(p + phase, axis=-1)


# This helper function is required as flax doesn't allow for unhashable attributes
def build_neural_mean_field(alphas_global, alphas, activation):
    """
    Build a neural mean field model for the classical degrees of freedom.
    Args:
        alphas_global: hidden units in the global layers
        alphas: hidden units in the real/imaginary
        activation: activation function
    Returns:
        NeuralMeanField: neural mean field model
    """

    class NeuralMeanField(nn.Module):
        """
        Mean field model for the classical degrees of freedom.
        The parameters depend on the quantum degrees of freedom through
        a dense layer.
        """

        @nn.compact
        def __call__(self, z_q, z_c):
            """
            Compute the log amplitudes for
            Args:
                z_q: quantum degrees of freedom
                z_c: classical degrees of freedom
            """
            n_classical = z_c.shape[-1]

            for alpha in alphas_global:
                z_q = nn.Dense(features=alpha * n_classical)(z_q)
                z_q = activation(z_q)
            lam_real = z_q
            lam_imag = z_q
            for alpha in alphas:
                lam_real = nn.Dense(features=alpha * n_classical)(lam_real)
                lam_real = activation(lam_real)
                lam_imag = nn.Dense(features=alpha * n_classical)(lam_imag)
                lam_imag = activation(lam_imag)

            lam = nn.Dense(
                features=z_c.shape[-1],
                name="real",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(lam_real)

            p = nn.log_sigmoid(lam * z_c)
            phase = (
                1j
                * nn.Dense(
                    features=z_c.shape[-1],
                    name="imag",
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.zeros,
                )(lam_imag)
                * z_c
            )
            return 0.5 * jnp.sum(p + phase, axis=-1)

    return NeuralMeanField()


class PureMeanField(nn.Module):
    """
    Mean field model for the classical degrees of freedom.
    The parameters are independent of the quantum degrees of freedom.
    """

    @nn.compact
    def __call__(self, z_q, z_c):
        """
        Compute the log amplitudes for
        Args:
            z_q: quantum degrees of freedom (not used)
            z_c: classical degrees of freedom
        """
        lam = self.param("real", nn.initializers.normal(), (z_c.shape[-1],), float)

        p = nn.log_sigmoid(lam * z_c)
        phase = (
            1j
            * self.param("imag", nn.initializers.normal(), (z_c.shape[-1],), float)
            * z_c
        )
        return 0.5 * jnp.sum(p + phase, axis=-1)


class QuantumPlusMeanField(nn.Module):
    """
    Wrapper combining the mean field model for the classical degrees of freedom with a model for the quantum degrees of freedom.
    """

    mf_model: nn.Module
    """Mean-field model for the classical partition"""
    quantum_model: nn.Module
    """Model for the quantum partition"""

    @nn.compact
    def __call__(self, z_q, z_c):
        """
        Compute the log amplitudes for
        Args:
            z_q: quantum degrees of freedom
            z_c: classical degrees of freedom
        """
        return self.mf_model(z_q, z_c) + self.quantum_model(z_q)


class NetketMeanField(nn.Module):
    """Wrapper around the `QuantumPlusMeanField` method for NetKet simulations"""

    qpmf_model: QuantumPlusMeanField
    """Quantum plus mean field model"""
    n_quantum: int
    """Number of quantum degrees of freedom"""

    @nn.compact
    def __call__(self, z):
        """
        Compute the log amplitudes for
        Args:
            z: quantum and classical degrees of freedom
        """
        z_q = z[:, : self.n_quantum]
        z_c = z[:, self.n_quantum :]
        return self.qpmf_model(z_q, z_c)


class TwoSubCQD:
    """
    CQD ansatz for two subsystems, where the quantum subsystem is treated with a model and the classical subsystem with a mean field model.
    """

    def __init__(self, model: QuantumPlusMeanField, n_classical: int, n_quantum: int):
        """
        Args:
            model: total model for the log-amplitudes of both subsystems
            n_classical: number of classical degrees of freedom
            n_quantum: number of quantum degrees of freedom
        """
        self.model = model
        self.n_classical = n_classical
        self.n_quantum = n_quantum
        self._gradient = jax.jacfwd(self.model.apply)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, theta: dict, z_q, z_c) -> jnp.ndarray:
        """
        Compute the logamplitudes for
        Args:
            theta: parameters of the model
            z_q: quantum degrees of freedom
            z_c: classical degrees of freedom
        """
        return jnp.exp(self.model.apply(theta, z_q, z_c))

    def q_amp(self, theta, z_q):
        """Return the amplitude square of the quantum part
        Args:
            theta: parameters of the model
            z_q: quantum degrees of freedom
        """
        return (
            jnp.abs(
                jnp.exp(
                    self.model.quantum_model.apply(
                        {"params": theta["params"]["quantum_model"]}, z_q
                    )
                )
            )
            ** 2
        )

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, theta, z):
        """
        Wrapper for the model apply method that takes a single input
        Args:
            theta: parameters of the model
            z: quantum and classical degrees of freedom
        """
        z_q = z[:, : self.n_quantum]
        z_c = z[:, self.n_quantum :]
        return self.model.apply(theta, z_q, z_c)

    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, theta: dict, z_q, z_c) -> jnp.ndarray:
        """
        Compute the gradient of the log amplitudes for the two subsystems
        shape: (n_q_samples, n_c_samples, n_params)
        Args:
            theta: parameters of the model
            z_q: quantum degrees of freedom
            z_c: classical degrees of freedom
        """
        q_shape = z_q.shape
        c_shape = z_c.shape
        z_q = z_q.reshape((-1, self.n_quantum))
        z_c = z_c.reshape((-1, self.n_classical))
        z = jnp.hstack([z_q, z_c])
        jac = nk.jax.jacobian(self.apply, theta["params"], z, mode="complex")
        # reshape final axis (if the parameters are matrices)
        jac = jax.tree.map(lambda x: x.reshape((x.shape[0], 2, -1)), jac)
        # Sum real and imagin
        jac = jax.tree.map(lambda x: x[:, 0, :] + 1j * x[:, 1, :], jac)
        jac = jax.tree.flatten(jac)[0]
        jac = jnp.concatenate(jac, axis=-1)
        return jac.reshape((q_shape[:-1]) + jac.shape[1:])

    @partial(jax.jit, static_argnums=(0,))
    def full_state(self, theta, psi_q):
        """Return the full state vector of the system
        Args:
            theta: parameters of the model
            psi_q: quantum state
        """
        z_c = -all_z(self.n_classical)
        return self.classical_state(theta) * jnp.repeat(psi_q, z_c.shape[0])

    @partial(jax.jit, static_argnums=(0,))
    def classical_state(self, theta):
        """Return the full state vector of the system without taking the quantum state into account.
        Args:
            theta: parameters of the model
        """
        z_q = -all_z(self.n_quantum)
        z_c = -all_z(self.n_classical)
        amps = self(
            theta,
            jnp.repeat(z_q, z_c.shape[0], axis=0),
            jnp.tile(z_c, (z_q.shape[0], 1)),
        )
        return amps

    @partial(jax.jit, static_argnums=(0,))
    def full_gradient(self, theta):
        """Returns the full jacobian of the state. Shape (2**n_qubits, n_params).
        Args:
            theta: parameters of the model
        """
        z_c = -all_z(self.n_classical)
        z_q = -all_z(self.n_quantum)
        return self.gradient(
            theta,
            jnp.repeat(z_q, z_c.shape[0], axis=0),
            jnp.tile(z_c, (z_q.shape[0], 1)),
        )

    def sample_z_c(self, theta, z_q, key, n_samples):
        """Iteratively sample the classical degrees of freedom
        Args:
            theta: parameters of the model
            z_q: quantum degrees of freedom
            key: random key
            n_samples: number of samples
        Returns:
            z_q: quantum degrees, shape (z_q.shape[0],n_samples)
            z_c: classical degrees, shape (z_q.shape[0], n_samples, n_classical)
        """
        z_q = jnp.atleast_2d(z_q)
        z_ones = jnp.ones((z_q.shape[0], self.n_classical))

        # make an array with -1 on the diagonal
        z_shift = jnp.ones(self.n_classical) - 2 * jnp.eye(self.n_classical)
        # copy for every z_q
        z_shift = jnp.tile(z_shift, (z_q.shape[0], 1))

        # amplitudes square for all ones
        p_ones = jnp.abs(self(theta, z_q, z_ones)) ** 2
        p_ones = jnp.repeat(p_ones, self.n_classical)
        # amplitudes square for single spin flips
        p_shift = (
            jnp.abs(self(theta, jnp.repeat(z_q, self.n_classical, axis=0), z_shift))
            ** 2
        )

        # These are the individual probabilites for each spin
        p = 1 / (1 + p_shift / p_ones)
        p = p.reshape((z_q.shape[0], self.n_classical))
        z_c = -1 + 2 * jax.random.bernoulli(key, p, (n_samples,) + p.shape)
        z_c = jnp.swapaxes(z_c, 0, 1)
        z_q = jnp.repeat(z_q, n_samples, axis=0).reshape((-1, n_samples, z_q.shape[-1]))

        return z_q, z_c
