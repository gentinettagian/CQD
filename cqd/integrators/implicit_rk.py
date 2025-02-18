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


import jax
import jax.numpy as jnp


from cqd.utils import get_structure, random_tree_like
from .rk import iEuler, iMidpoint, imp_RKstep


class ImpRKIntegrator:
    """Class implementing implicit Runge-Kutta integrators using the netket.experimental interface"""

    def __init__(self, dt, B, rhs, t0, y0, key):
        """
        Args:
            dt: time step
            B: Butcher tableau
            rhs: right-hand side of the ODE
            t0: initial time
            y0: initial state
            key: PRNG key
        """
        self.dt = dt
        self.B = B
        self.rhs = rhs
        self.t = t0
        self.y = y0
        self.shapes, self.split_points, self.tree_struct = get_structure(y0)
        self.key = key
        self.steps = 0

    def step(self):
        """
        Perform a single step of the integrator
        """
        key, self.key = jax.random.split(self.key)
        guess = random_tree_like(self.y, key, 1)  # 0.1 ** (self.steps / 10))
        guess = jax.tree.map(lambda x, y: x + y, self.y, guess)
        y1 = imp_RKstep(
            self.B,
            self._rhs,
            self._pytree_to_array(self.y),
            self.t,
            self.dt,
            guess=self._pytree_to_array(guess),
        )
        self.y = self._array_to_pytree(y1)
        self.t += self.dt
        self.steps += 1

    def _rhs(self, t, y):
        rhs = self.rhs(t, self._array_to_pytree(y))
        return self._pytree_to_array(rhs)

    def _pytree_to_array(self, pytree):
        arrs = jax.tree.flatten(pytree)[0]
        arrs = [arr.flatten() for arr in arrs]
        return jnp.concatenate(arrs)

    def _array_to_pytree(self, arr):
        arr = jnp.split(arr, self.split_points)
        arr = [x.reshape(s) for x, s in zip(arr, self.shapes)]
        return jax.tree.unflatten(self.tree_struct, arr)


class ImpRKConfig:
    """Factory to create implicit Runge-Kutta integrators"""

    def __init__(self, dt, B, key):
        """
        Args:
            dt: time step
            B: Butcher tableau
            key: PRNG key
        """
        self.dt = dt
        self.B = B
        self.key = key

    def __call__(self, rhs, t0, y0):
        """Returns an implicit Runge-Kutta integrator with the provided ODE
        Args:
            rhs: right-hand side of the ODE
            t0: initial time
            y0: initial state
        """
        return ImpRKIntegrator(self.dt, self.B, rhs, t0, y0, self.key)


def ImplicitEuler(dt, key):
    """
    Returns an implicit Euler integrator.
    """
    return ImpRKConfig(dt, iEuler, key)


def ImplicitMidpoint(dt, key):
    """
    Returns an implicit midpoint integrator.
    """
    return ImpRKConfig(dt, iMidpoint, key)
