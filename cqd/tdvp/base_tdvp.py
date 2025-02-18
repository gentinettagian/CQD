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
from abc import abstractmethod


import jax.numpy as jnp
import jax
from netket.experimental.dynamics import RKIntegratorConfig, Euler

from cqd.utils import *
from cqd.expectation import PauliSum
from cqd.integrators import ImpRKConfig
from .trotter import TrotterTerm
from .pinv_smooth import pinv_smooth


class BaseTDVP:
    """Base class for the TDVP with the CQD framework."""

    def __init__(
        self,
        initial_parameters: dict,
        init_quantum_state: jnp.ndarray,
        integrator: RKIntegratorConfig | ImpRKConfig = None,
        dt: float = 0.01,
        acond=1e-5,
        rcond=1e-4,
        h_tilde: PauliSum | None = None,
        trotter_step: float | None = None,
        trotter_terms: list[TrotterTerm] | None = None,
        correct_trotter: bool = True,
        shots: int | None = None,
    ):
        """
        Args:
            initial_parameters: Initial parameters of the system.
            init_quantum_state: Initial quantum state.
            integrator: Integrator to use for the time evolution.
            dt: Time step for the integrator (if integrator is provided this is ignored).
            acond, rcond: Regularization parameters for the pseudo-inverse.
            h_tilde: The tilde Hamiltonian.
            trotter_step: The time step for the trotterization. Set to `None` for exact evolution.
            trotter_terms: The terms in the trotterization.
            correct_trotter: Whether to correct the trotterization.
            shots: Number of shots for the expectation value estimation. Set to `None` for exact evaluation.
        """

        self.h_tilde = h_tilde

        if integrator is None:
            integrator = Euler(dt)
        self.integrator = integrator
        self.acond = acond
        self.rcond = rcond
        self.theta0 = initial_parameters
        self.shapes, self.split_points, self.tree_struct = get_structure(
            initial_parameters
        )

        self.trotter_step = trotter_step
        self.correct_trotter = correct_trotter
        if trotter_step is not None:
            self.trotter_terms = trotter_terms
            self.total_step_time = sum([term.time_fraction for term in trotter_terms])
        self.shots = shots

        # Initial quantum state
        if init_quantum_state is None:
            self.phi_q = jnp.ones(h_tot.shape[0], dtype=jnp.complex128) / jnp.sqrt(
                2**self.n_spins
            )
        else:
            self.phi_q = init_quantum_state
        self.t = 0.0

        if trotter_step:
            if correct_trotter:

                def h_tilde():
                    return self.current_trotter_term()[0].paulis * self.total_step_time

            self.h_tilde = h_tilde
        else:
            self.D_tilde, self.V_tilde = jnp.linalg.eigh(h_tilde.to_dense())
            self.h_tilde = h_tilde

        self.inverter = jax.jit(
            lambda g, b: pinv_smooth(
                jnp.real(g), jnp.imag(b), rcond_smooth=self.acond, rcond=self.rcond
            )[0]
        )
        self._run_info = []

    def current_trotter_term(self):
        """Returns the trotter term currently acting on the quantum state as well as the remaining time in the current step."""
        if self.trotter_step is None:
            return None
        t_ = jnp.round(self.t % self.trotter_step, 8)
        term_ind = 0
        while (
            t_
            >= self.trotter_terms[term_ind % len(self.trotter_terms)].time_fraction
            * self.trotter_step
            / self.total_step_time
        ):
            t_ -= (
                self.trotter_terms[term_ind % len(self.trotter_terms)].time_fraction
                * self.trotter_step
                / self.total_step_time
            )
            t_ = jnp.round(t_, 8)
            term_ind += 1
        trotter_term = self.trotter_terms[term_ind % len(self.trotter_terms)]
        return (
            trotter_term,
            trotter_term.time_fraction * self.trotter_step / self.total_step_time - t_,
        )

    def evolve_quantum_state(self, dt: float):
        """Evolve the quantum state by dt."""
        if self.trotter_step is None:
            self.phi_q = exact_evolve(self.V_tilde, self.D_tilde, self.phi_q, dt)
            self.t += dt
            return
        current_step, remaining_time = self.current_trotter_term()
        if remaining_time >= dt:
            self.phi_q = trotter_evolve(
                current_step, self.phi_q, dt * self.total_step_time
            )
            self.t += dt
        else:
            self.phi_q = trotter_evolve(
                current_step, self.phi_q, remaining_time * self.total_step_time
            )
            dt -= remaining_time
            self.t += remaining_time
            self.evolve_quantum_state(dt)

    def run(self, end_time: float, callback=None):
        """Run the simulation for end_time.
        Args:
            end_time: The time to run the simulation for.
            callback: A function that will be called at each time step with the current time and state.
                has the signature `callback(time, theta, self)`.
        """
        if self.integrator is None:
            raise ValueError("No integrator provided.")
        integrator = self.integrator(self._odefun, 0.0, self.theta0)
        while integrator.t < end_time:
            integrator.step()
            self.evolve_quantum_state(integrator.t - self.t)
            if callback is not None:
                cancel = callback(
                    integrator.t,
                    integrator.y,
                    self,
                )
                if cancel:
                    break

        return integrator.y

    @abstractmethod
    def exact_state(self, t: float):
        """Returns the exact quantum state at time t."""
        return NotImplementedError

    @abstractmethod
    def _forces_and_qgt(self, t: float, theta: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
        return NotImplementedError

    def _odefun(self, t: float, theta: dict, *args, **kwargs):
        # Handle cases for implicit solvers
        if jnp.round(t, 8) > jnp.round(self.t, 8):
            old_t = self.t
            old_phi_q = self.phi_q
            self.evolve_quantum_state(t - old_t)
        elif jnp.round(t, 8) < jnp.round(self.t, 8):
            raise ValueError("Time must be increasing")
        else:
            old_phi_q = self.phi_q
            old_t = self.t
        # Evaluate forces at the new time
        b, g = self._forces_and_qgt(t, theta)
        # Reset the state
        self.t = old_t
        self.phi_q = old_phi_q

        thetadot = self.inverter(g, b)
        thetadot = jnp.split(thetadot, self.split_points)
        thetadot = [x.reshape(s) for x, s in zip(thetadot, self.shapes)]
        thetadot = jax.tree.unflatten(self.tree_struct, thetadot)
        return thetadot


@jax.jit
def exact_evolve(V, D, phi0, t):
    return V @ jnp.diag(jnp.exp(-1.0j * D * t)) @ jnp.conj(V.T) @ phi0


# @partial(jax.jit, static_argnums=(0,))
def trotter_evolve(trotter_term: TrotterTerm, phi0: jnp.ndarray, t: float):
    instructions = trotter_term.evolve_instructions(t)
    for instr in instructions:
        phi0 = instr.dot(phi0)
    return phi0
