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
import jax.scipy as jsp

from netket.jax import tree_ravel
from netket.utils.api_utils import partial_from_kwargs


@partial_from_kwargs
def pinv_smooth(A, b, *, rcond: float = 1e-14, rcond_smooth: float = 1e-14, x0=None):
    r"""
    Solve the linear system by building a pseudo-inverse from the
    eigendecomposition obtained from :func:`jax.numpy.linalg.eigh`.

    The eigenvalues :math:`\lambda_i` smaller than
    :math:`r_\textrm{cond} \lambda_\textrm{max}` are truncated (where
    :math:`\lambda_\textrm{max}` is the largest eigenvalue).

    The eigenvalues are further smoothed with another filter, originally introduced in
    `Medvidovic, Sels arXiv:2212.11289 (2022) <https://arxiv.org/abs/2212.11289>`_,
    given by the following equation

    .. math::

        \tilde\lambda_i^{-1}=\frac{\lambda_i^{-1}}{1+\big(\epsilon\frac{\lambda_\textrm{max}}{\lambda_i}\big)^6}


    .. note::

        In general, we found that this custom implementation of
        the pseudo-inverse outperform
        jax's :func:`~jax.numpy.linalg.pinv`. This might be
        because :func:`~jax.numpy.linalg.pinv` internally calls
        :obj:`~jax.numpy.linalg.svd`, while this solver internally
        uses :obj:`~jax.numpy.linalg.eigh`.

        For that reason, we suggest you use this solver instead of
        :obj:`~netket.optimizer.solver.pinv`.


    .. note::

        If you pass only keyword arguments, this solver will directly create
        a partial capturing them.


    Args:
        A: LinearOperator (matrix)
        b: vector or Pytree
        rcond : Cut-off ratio for small singular values of :code:`A`. For
            the purposes of rank determination, singular values are treated
            as zero if they are smaller than rcond times the largest
            singular value of :code:`A`.
        rcond_smooth : regularization parameter used with a similar effect to `rcond`
            but with a softer curve. See :math:`\epsilon` in the formula
            above.
    """
    del x0

    if not isinstance(A, jax.Array):
        A = A.to_dense()
    b, unravel = tree_ravel(b)

    Σ, U = jnp.linalg.eigh(A)

    # Discard eigenvalues below numerical precision
    Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rcond, jnp.reciprocal(Σ), 0.0)

    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(Σ / Σ[-1])) ** 6)

    Σ_inv = Σ_inv * regularizer

    x = U @ (Σ_inv * (U.conj().T @ b))

    return unravel(x), None
