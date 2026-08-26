"""
Random linear mode coupling for multimode fiber: power/phase exchange
between principal mode groups from real-world perturbations (bends,
splices, microbending) -- a mechanism distinct from, and additional to,
everything in fiber.multimode_propagator.MultimodeFiberPropagator, which
only couples mode groups DETERMINISTICALLY through nonlinearity (Kerr
XPM, Raman). Real fiber ALSO exchanges power LINEARLY between mode
groups from random imperfections along its length, with no nonlinearity
or intensity dependence involved -- this is the mechanism that, among
other things, determines how much of a qubit launched purely into one
mode group leaks into others over a real link, independent of any
nonlinear effect.

Physics
-------
Modelled as a random unitary rotation among mode groups, applied once
per propagation step from a random HERMITIAN matrix K(z):

    U(z) = expm(i*K(z))
    K_pq(z) ~ CN(0, kappa^2*dz) * overlap[p,q]   (Hermitian: K_qp = conj(K_pq))

kappa (rad/sqrt(m)) sets the coupling strength; overlap[p,q] is the same
spatial-overlap-decay matrix MultimodeFiberParams already computes for
the nonlinear coupling model (mode groups closer in index are expected
to couple more strongly under smooth, gradual perturbations like
bending -- the same physical intuition, reused rather than re-modelled).

Being generated from an exactly Hermitian K, U(z) is exactly unitary at
every step, so this term conserves total power across mode groups
exactly regardless of coupling strength (a lossless redistribution --
verified in tests/test_mode_coupling.py) -- and, being an independent
random draw at each step, the ACCUMULATED coupling after distance L
grows as kappa*sqrt(L) (diffusive/random-walk scaling), not linearly
with L -- consistent with how strong-mode-coupling and PMD accumulation
are described in the multimode-fiber and space-division-multiplexing
literature.

Approximation
-------------
kappa has no standard reference figure the way OFL/EMB bandwidth does
for DMD or dma_fraction does (loosely) for differential mode
attenuation; there is no widely agreed "typical OM3 mode-coupling
length" the way there's a nominal bandwidth spec. Calibrate kappa
against a measured coupling length for your specific fiber/cabling if
you have one; the default here is a representative order-of-magnitude
placeholder, not a datasheet value.

    from fiber.mode_coupling import RandomModeCouplingPropagator
"""
import numpy as np
from scipy.linalg import expm

from fiber.multimode_propagator import MultimodeFiberPropagator


class RandomModeCouplingPropagator(MultimodeFiberPropagator):
    """MultimodeFiberPropagator with random linear mode coupling added per step.

    Parameters
    ----------
    fiber : fiber.multimode_fiber.MultimodeFiberParams
    kappa : float
        Coupling strength (rad/sqrt(m)). Representative default; see
        module docstring's Approximation note. Larger kappa -> shorter
        effective mode-coupling length -> faster power redistribution
        among mode groups.
    seed : int or None -- RNG seed for a reproducible coupling realization
    include_raman : bool -- as in MultimodeFiberPropagator
    """

    def __init__(self, fiber, kappa=0.05, seed=None, include_raman=True):
        super().__init__(fiber, include_raman=include_raman)
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

    def _linear_coupling_step(self, A_t, dz):
        M = self.fiber.n_modes
        std = self.kappa * np.sqrt(dz) / np.sqrt(2)
        raw = (self.rng.standard_normal((M, M)) + 1j * self.rng.standard_normal((M, M))) * std
        K = (raw + raw.conj().T) / 2          # exactly Hermitian
        K = K * self.fiber.overlap             # weight by the existing spatial-overlap model
        K = (K + K.conj().T) / 2               # re-symmetrize (overlap is real & symmetric,
                                                # but guard against float round-off asymmetry)
        U = expm(1j * K)
        return U @ A_t
