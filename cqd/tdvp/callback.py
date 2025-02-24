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

import numpy as np
import jax.numpy as jnp

from cqd.expectation import PauliSum, PauliString
from cqd.utils import all_z, zero_tree_like

from .base_tdvp import exact_evolve
from .hybrid_tdvp import HybridTDVP


class CQDCallback:
    """Creates a default callback that computes fidelities and expectation values"""

    def __init__(
        self,
        h_tot: PauliSum,
        init_state: jnp.ndarray,
        observables: list[PauliSum | PauliString] | None = None,
        compute_exact: bool = True,
        verbose: bool = True,
    ):
        """Args:
        h_tot: The total Hamiltonian of the system
        init_state: The initial quantum state of the system
        observables: The observables to compute expectation values for
        compute_exact: If set to false, only expectation values will be computed, no exact solution is provided
        verbose: If set to true, prints the fidelities at each step
        """
        self.h_tot = h_tot
        if observables is None:
            observables = []
        self.observables = observables
        self.compute_exact = compute_exact
        self.verbose = verbose
        self._fidelities = []
        self._expectation_values = []
        self._times = []

        if compute_exact:
            self.allz = all_z(h_tot.n_qubits)
            self.d, self.v = jnp.linalg.eigh(h_tot.to_dense())
            self.psi0 = init_state.copy()
            self.sparse_observables = [obs.to_sparse() for obs in observables]

    def __call__(self, t: float, theta: dict, tdvp: HybridTDVP):
        """Callback function that computes fidelities and expectation values"""
        self._times.append(t)
        expvals = [tdvp.expect(obs, theta) for obs in self.observables]
        exvals_q = [tdvp.expect(obs, zero_tree_like(theta)) for obs in self.observables]
        if self.compute_exact:
            state = tdvp.phi_q * tdvp.model(theta, self.allz)
            state /= jnp.linalg.norm(state)
            exact = exact_evolve(self.v, self.d, self.psi0, t)
            fidelity = jnp.abs(jnp.vdot(exact, state)) ** 2
            fid_q = jnp.abs(jnp.vdot(exact, tdvp.phi_q)) ** 2
            self._fidelities.append([fidelity, fid_q])
            if self.verbose:
                print(
                    f"Time {t:.3f}: Fidelity {fidelity:.3f}, Fidelity q {fid_q:.3f}",
                    end="\r",
                )
            expvals_exact = [
                jnp.vdot(exact, obs.dot(exact)) for obs in self.sparse_observables
            ]
            self._expectation_values.append([expvals, expvals_exact, exvals_q])
        else:
            self._expectation_values.append([expvals, exvals_q])
            if self.verbose:
                print(
                    f"Time {t:.3f}: Expectation values {expvals}",
                    end="\r",
                )

    @property
    def fidelities(self):
        return np.array(self._fidelities)

    @property
    def expectation_values(self):
        return np.array(self._expectation_values)

    @property
    def times(self):
        return np.array(self._times)
