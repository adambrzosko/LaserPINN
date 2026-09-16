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
    from fiber.mode_coupling import RandomModeCouplingPropagator
    from fiber.quantum_multimode import QuantumMultimodePropagator, ensemble_propagate_multimode
    from fiber.wdm_propagator import WDMPropagator
    from fiber.quantum_wdm import QuantumWDMPropagator, ensemble_propagate_wdm
    from fiber.four_wave_mixing import fwm_efficiency, fwm_power, fwm_ghost_tone_power
    from fiber.polarization import PolarizationPropagator
    from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power
    from fiber.rayleigh_backscatter import rayleigh_backscatter_power, rayleigh_otdr_trace
    from fiber.hybrid_crosstalk import HybridCrosstalkPropagator
    from fiber.receiver_leakage import filter_leakage_photons, required_floor_dB
    from fiber.raman_models import BlowWood, LinAgrawal
    from fiber.grin_modes import DESIGNS, FibreDesign, FibreModes
    from fiber.gmmnlse import GMMNLSE, ModeCoupling, ModeLoss, PropagationResult, TimeGrid

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

For random LINEAR mode coupling (bends/splices, distinct from the
nonlinear coupling above):

    from fiber.mode_coupling import RandomModeCouplingPropagator

    rc_prop = RandomModeCouplingPropagator(om4, kappa=0.05, seed=0)
    A_out = rc_prop.propagate(A0, dt, L=200)

For the spontaneous-Raman noise floor a bright signal in one mode group
imposes on itself and on other (possibly empty) mode groups:

    from fiber.quantum_multimode import QuantumMultimodePropagator

    qmm_prop = QuantumMultimodePropagator(om4, seed=0)
    A_out = qmm_prop.propagate(A0, dt, L=200)

For multiple co-propagating WDM channels sharing one spatial mode, with
self-phase modulation, cross-phase modulation, and inter-channel Raman
crosstalk:

    from fiber.wdm_propagator import WDMPropagator

    wdm = WDMPropagator(smf, channel_offsets_Hz=[-100e9, 0.0, 100e9])
    A_out = wdm.propagate(A0, dt, L=10e3)   # A0 shape (3, n_pts) or launched into channel 0

For the spontaneous-Raman noise floor a bright WDM channel imposes on a
weak co-propagating one (e.g. a classical data channel next to a quantum
channel):

    from fiber.quantum_wdm import QuantumWDMPropagator

    qwdm = QuantumWDMPropagator(smf, channel_offsets_Hz=[0.0, -13.2e12], seed=0)
    A_out = qwdm.propagate(A0, dt, L=10e3)

For four-wave mixing (an analytical, undepleted-pump ghost-tone power
calculation, not a dynamical propagator -- see fiber/four_wave_mixing.py):

    from fiber.four_wave_mixing import fwm_ghost_tone_power

    result = fwm_ghost_tone_power(smf, channel_powers_W=[0.01, 0.01, 0.01],
                                   channel_offsets_Hz=[-100e9, 0.0, 100e9],
                                   target_offset_Hz=0.0, L=10e3)
    print(result['total_power_W'], result['contributions'])

For polarization mode dispersion (PMD) and polarization-dependent XPM/
Raman (a 2-component Jones-vector propagator):

    from fiber.polarization import PolarizationPropagator

    pol_prop = PolarizationPropagator(smf, D_PMD=0.1, seed=0)  # ps/sqrt(km)
    A_out = pol_prop.propagate(A0, dt, L=10e3)  # A0 shape (2, n_pts) or launched into x

For stimulated Brillouin scattering (steady-state coupled pump/backward-
Stokes power equations, solved as a boundary value problem):

    from fiber.brillouin import BrillouinPropagator, sbs_threshold_power

    P_th = sbs_threshold_power(smf, L=20e3)
    bp = BrillouinPropagator(smf)
    z, P_pump, P_stokes = bp.solve(P_pump_in=2 * P_th, L=20e3)

For elastic Rayleigh backscattering (linear, always-present, distinct
from SBS -- the physics behind OTDR):

    from fiber.rayleigh_backscatter import rayleigh_backscatter_power

    P_back = rayleigh_backscatter_power(smf, P_in=0.001, L=20e3)

For a bright classical channel and a weak signal separated by BOTH
spatial mode AND DWDM wavelength at once (e.g. a bright reference in one
mode group on one channel, phase-encoded QKD in another mode group on
another channel):

    from fiber.multimode_fiber import make_multimode_fiber
    from fiber.hybrid_crosstalk import HybridCrosstalkPropagator

    om3 = make_multimode_fiber('om3', lambda0=1552.524e-9)
    hybrid = HybridCrosstalkPropagator(om3, qkd_mode=0, bright_mode=1,
                                        channel_separation_Hz=200e9, noise=True, seed=0)
    A_qkd_out, A_bright_out = hybrid.propagate(A_qkd_0, A_bright_0, dt, L=10e3)

For estimating direct classical-carrier leakage through a finite-
rejection receive filter (a distinct, linear, post-fiber mechanism from
the in-fiber crosstalk above -- often the dominant one in practice):

    from fiber.receiver_leakage import filter_leakage_photons

    n_leaked = filter_leakage_photons(P_bright_launch_W=1e-3, L=10e3, fiber=om3,
                                       bright_mode=1, filter_floor_dB=40.0,
                                       omega_qkd=om3.omega0, gate_window_s=500e-12)
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
from fiber.mode_coupling import RandomModeCouplingPropagator
from fiber.quantum_multimode import QuantumMultimodePropagator, ensemble_propagate_multimode
from fiber.wdm_propagator import WDMPropagator
from fiber.quantum_wdm import QuantumWDMPropagator, ensemble_propagate_wdm
from fiber.four_wave_mixing import fwm_efficiency, fwm_power, fwm_ghost_tone_power
from fiber.polarization import PolarizationPropagator
from fiber.brillouin import BrillouinPropagator, sbs_threshold_power, spontaneous_brillouin_noise_power
from fiber.rayleigh_backscatter import rayleigh_backscatter_power, rayleigh_otdr_trace
from fiber.hybrid_crosstalk import HybridCrosstalkPropagator
from fiber.receiver_leakage import filter_leakage_photons, required_floor_dB
from fiber.raman_models import BlowWood, LinAgrawal
from fiber.grin_modes import DESIGNS, FibreDesign, FibreModes
from fiber.gmmnlse import GMMNLSE, ModeCoupling, ModeLoss, PropagationResult, TimeGrid
