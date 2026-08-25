"""
fiber -- Nonlinear fiber propagation toolkit (geometries, materials, and
classical/quantum Raman scattering), built to consume the field output of
core.dfb_laser / core.million_pulse_comparison / core.sld_injection.

    from fiber.materials import FiberMaterial, make_material
    from fiber.geometry import FiberGeometry, make_geometry
    from fiber.fiber_params import FiberParams, make_fiber
    from fiber.propagator import FiberPropagator
    from fiber.quantum_noise import QuantumRamanPropagator, ensemble_propagate
    from fiber.sources import intracavity_to_field, extract_pulse, zero_pad
    from fiber.analysis import pulse_metrics, spectral_centroid, band_power
    from fiber.multimode_fiber import MultimodeFiberGeometry, MultimodeFiberParams, make_multimode_fiber
    from fiber.multimode_propagator import MultimodeFiberPropagator
    from fiber.wdm_propagator import WDMPropagator
    from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power

Quick start
-----------
    from fiber.fiber_params import make_fiber
    from fiber.propagator import FiberPropagator

    smf = make_fiber('smf28')
    prop = FiberPropagator(smf)
    A_out = prop.propagate(A0, dt, L=10e3)   # 10 km

For spontaneous-Raman quantum noise on top of the classical field:

    from fiber.quantum_noise import QuantumRamanPropagator
    qprop = QuantumRamanPropagator(smf, seed=0)
    A_out = qprop.propagate(A0, dt, L=10e3)

For multimode (OM1-OM5) graded-index fiber, with intermodal dispersion
and intermodal Raman scattering between principal mode groups:

    from fiber.multimode_fiber import make_multimode_fiber
    from fiber.multimode_propagator import MultimodeFiberPropagator

    om4 = make_multimode_fiber('om4')
    mm_prop = MultimodeFiberPropagator(om4)
    A_out = mm_prop.propagate(A0, dt, L=200)   # A0 launched into mode group 0

For multiple co-propagating WDM channels sharing one spatial mode, with
self-phase modulation, cross-phase modulation, and inter-channel Raman
crosstalk:

    from fiber.wdm_propagator import WDMPropagator

    wdm = WDMPropagator(smf, channel_offsets_Hz=[-100e9, 0.0, 100e9])
    A_out = wdm.propagate(A0, dt, L=10e3)   # A0 shape (3, n_pts) or launched into channel 0

For stimulated Brillouin scattering (steady-state coupled pump/backward-
Stokes power equations, solved as a boundary value problem):

    from fiber.brillouin import BrillouinPropagator, sbs_threshold_power

    P_th = sbs_threshold_power(smf, L=20e3)
    bp = BrillouinPropagator(smf)
    z, P_pump, P_stokes = bp.solve(P_pump_in=2 * P_th, L=20e3)
"""
from fiber.materials import FiberMaterial, make_material
from fiber.geometry import FiberGeometry, make_geometry
from fiber.fiber_params import FiberParams, make_fiber
from fiber.propagator import FiberPropagator
from fiber.quantum_noise import QuantumRamanPropagator, ensemble_propagate
from fiber.sources import intracavity_to_field, extract_pulse, zero_pad
from fiber.analysis import pulse_metrics, spectral_centroid, band_power
from fiber.multimode_fiber import MultimodeFiberGeometry, MultimodeFiberParams, make_multimode_fiber
from fiber.multimode_propagator import MultimodeFiberPropagator
from fiber.wdm_propagator import WDMPropagator
from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power
