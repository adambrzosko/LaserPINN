# Gain-Switched DFB Laser Simulation Suite

## Overview

This simulation suite models the dynamics of a 1550 nm InGaAsP/InP gain-switched DFB laser for quantum key distribution (QKD) source characterisation. The core physics solves stochastic rate equations (carrier density N, photon density S, optical phase phi) using Euler-Maruyama integration with Langevin noise sources, compiled via Numba JIT for performance.

The primary metric of interest is the **inter-pulse phase correlation r1**, which quantifies residual coherence between consecutive pulses and directly impacts QKD security.

---

## Architecture

```
dfb_laser.py                  Core parameters (DFBLaserParams dataclass) + make_laser() factory
                              Supports DFB (R2=0.95 HR) and Fabry-Perot (R1=R2=0.32) cavities
sld_injection.py              SLD broadband injection model (ASE coupling)
million_pulse_comparison.py   High-throughput stochastic solver (Numba JIT) + waveform builders
├── simulate_pulses()              Standard raised-cosine modulation
├── simulate_pulses_waveform()     Arbitrary I(t) waveform input
├── build_raised_cosine()          Default modulation shape
├── build_square()                 Ideal step pulse
├── build_gaussian()               Gaussian envelope
├── build_fourier()                Fourier-parameterised envelope
└── build_trapezoid()              Asymmetric trapezoid (variable rise/fall)
```

All analysis scripts import from these core modules.

`fiber/` is a separate, independent package for nonlinear fiber propagation, built to consume the complex field output of `core.dfb_laser` / `core.million_pulse_comparison` / `core.sld_injection`:

```
fiber/materials.py       FiberMaterial dataclass + make_material(); Raman response (Blow-Wood damped-
                          oscillator model), phonon occupation (Bose-Einstein)
fiber/geometry.py        FiberGeometry dataclass + make_geometry(); sets A_eff (drives gamma).
                          smf28 A_eff is 85 um^2 (Corning's datasheet at 1550 nm, and what
                          grin_modes computes independently: 85.4); it was 80e-12, a 6% error in
                          gamma for every legacy-stack result. NA 0.14 is the far-field datasheet
                          figure -- the equivalent index-step NA is 0.115, as used by grin_modes.
fiber/fiber_params.py    FiberParams dataclass + make_fiber('smf28'|'dcf'|'hnlf'|
                          'pcf_supercontinuum'|'chalcogenide_waveguide') factory; combines
                          material+geometry+published D/alpha/beta3 into derived alpha/beta2/gamma
fiber/raman_response.py  Analytic frequency-domain Raman response H_R(Omega); classical stimulated-
                          Raman gain spectrum g_R(Omega). This is the single-oscillator Blow-Wood
                          model (f_R = 0.18) used by every legacy propagator; raman_models.LinAgrawal
                          (f_R = 0.245, with the boson peak) is what gmmnlse uses. They agree to 12%
                          at the 13.2 THz peak but not close in: Lin-Agrawal gives 1.4x more gain at
                          0.39 THz and 2.4x at 1 THz, where closely spaced WDM/QKD channels sit.
fiber/propagator.py      FiberPropagator: symmetric split-step GNLSE solver (dispersion + Kerr SPM +
                          full time-domain Raman response); f_R=0 recovers the plain-Kerr NLSE
fiber/quantum_noise.py   QuantumRamanPropagator(FiberPropagator): adds spontaneous-Raman Langevin
                          noise per step, strength set by Bose-Einstein phonon occupation
                          (fluctuation-dissipation) -- semiclassical quantum noise on top of the
                          classical field; ensemble_propagate() for multi-realization noise stats
fiber/sources.py          intracavity_to_field(), extract_pulse(), zero_pad() -- adapters from
                          core.dfb_laser-style intracavity fields to a fiber launch field
fiber/analysis.py         pulse_metrics(), spectral_centroid() (tracks soliton self-frequency
                          shift), band_power()
fiber/multimode_fiber.py  MultimodeFiberGeometry + MultimodeFiberParams dataclasses +
                          make_multimode_fiber('om1'..'om5') factory; reduced-order principal-
                          mode-group model (not full per-LP-mode) -- intermodal (DMD) group delay
                          per mode group via a calibrated graded-index alpha-profile formula, plus
                          a spatial-overlap-decay model for intermodal Kerr/Raman coupling strength.
                          alpha, beta2, and beta3 are ALL per-mode arrays, not scalars: alpha carries
                          differential mode attenuation (power-law growth with mode index, calibrated
                          per OM grade); beta2/beta3 carry a mode-dependent waveguide-dispersion
                          correction derived by finite-differencing the SAME delay-vs-frequency
                          formula used for DMD (beta2=d(beta1)/d(omega), beta3=d^2(beta1)/d(omega^2))
                          rather than new hand-tuned constants. Every per-mode quantity is referenced
                          to mode 0, so a single-mode-only launch reproduces FiberPropagator exactly.
                          Also derives delta_beta0 (M,): each mode group's ABSOLUTE propagation-
                          constant offset (rad/m), via the same leading-order WKB alpha-profile
                          relation as beta1 -- since beta1 IS beta0's frequency derivative by
                          definition, this is beta0's closed-form antiderivative, not a new model
                          (verified in tests/test_multimode_fiber.py: differentiating it numerically
                          reproduces beta1's leading term). Needed to correctly interfere/combine two
                          mode groups (e.g. a phase-encoded signal in one mode with a reference field
                          in another) -- delta_beta1 alone only governs each mode's own envelope.
fiber/multimode_propagator.py  MultimodeFiberPropagator: symmetric split-step solver for coupled
                          mode-group envelopes -- per-mode loss, GVD, TOD, and intermodal walk-off
                          (DMD) (reduces exactly to FiberPropagator when only one mode group is
                          populated), intramodal Kerr+Raman, intermodal Kerr XPM + intermodal
                          Raman coupling between mode groups, and (applied once, in closed form, to
                          the final output) each mode's absolute delta_beta0*L phase
fiber/mode_coupling.py   RandomModeCouplingPropagator(MultimodeFiberPropagator): random LINEAR
                          mode coupling from real-world perturbations (bends, splices) -- distinct
                          from, and additional to, the deterministic nonlinear coupling above. Each
                          step applies a random unitary rotation among mode groups, built from an
                          exactly Hermitian random matrix (so power is conserved exactly regardless
                          of coupling strength -- verified in tests/test_mode_coupling.py to 1e-15),
                          weighted by the same overlap-decay matrix used for nonlinear coupling
                          (nearby mode groups couple far more strongly than distant ones). The
                          accumulated coupling grows as kappa*sqrt(L) (diffusive/random-walk
                          scaling, matching how strong mode coupling and PMD accumulation are
                          described in the SDM/multimode-fiber literature), not linearly with L.
fiber/quantum_multimode.py  QuantumMultimodePropagator(MultimodeFiberPropagator): the multimode
                          counterpart of quantum_noise.QuantumRamanPropagator/quantum_wdm.
                          QuantumWDMPropagator -- adds spontaneous Raman noise, both intramodal
                          (mode 0's formula verified to exactly match the single-mode case) and
                          intermodal (weighted by the same gamma_matrix used for classical
                          intermodal coupling). Key physical difference from the classical
                          intermodal term: spontaneous noise here is driven by each mode's LOCAL
                          PEAK power directly, so unlike the deterministic term (which needs
                          genuine time-varying power and gives exactly zero for a CW driver -- see
                          multimode_propagator.py's docstring), a quasi-CW signal in one mode group
                          DOES seed measurable spontaneous noise in other, initially-empty mode
                          groups, with locality (nearby >> distant) matching the overlap-decay
                          weighting -- the mechanism relevant to a bright classical channel sharing
                          a multimode fiber with a weak quantum channel in a different spatial mode.
fiber/wdm_propagator.py  WDMPropagator: symmetric split-step solver for N co-propagating WDM
                          channels sharing one spatial mode -- per-channel walk-off from chromatic
                          dispersion, intra-channel SPM+Raman (reduces exactly to FiberPropagator
                          for 1 channel), instantaneous-Kerr cross-phase modulation (XPM, factor
                          2 vs SPM) between channels, and inter-channel Raman scattering ("Raman
                          crosstalk"/"Raman tilt") using the Raman gain spectrum evaluated at each
                          channel PAIR's fixed carrier separation -- a real power gain/loss term
                          that (unlike XPM or the multimode intermodal-Raman term) works even for
                          unmodulated/CW channels. Also derives delta_beta0 (N,): each channel's
                          ABSOLUTE propagation-constant offset, via fiber.beta1_ref (=material.n_g/c,
                          a new explicit absolute-index parameter -- everything else in this codebase
                          only ever needed relative dispersion) plus the exact antiderivative of
                          delta_beta1; applied once, in closed form, to the final propagated output.
                          Needed for phase-encoded protocols where the channel of interest will be
                          coherently interfered with something on a different wavelength (e.g. a
                          local oscillator, or a twin/reference pulse) -- not needed if co-propagating
                          channels only ever interact through the power-domain mechanisms above (XPM,
                          Raman crosstalk) and are never combined interferometrically.
fiber/quantum_wdm.py     QuantumWDMPropagator(WDMPropagator): the WDM counterpart of
                          quantum_noise.QuantumRamanPropagator -- adds BOTH intra-channel spontaneous
                          Raman noise (same physics as the single-mode case, per channel) AND
                          inter-channel spontaneous Raman noise (the quantum-noise counterpart of
                          WDMPropagator's deterministic Raman-crosstalk term: a bright channel's
                          spontaneous Raman scattering landing in a weak co-propagating channel's
                          slot -- the noise mechanism most often cited as dominant in classical/
                          quantum coexistence on shared fiber). ensemble_propagate_wdm() for multi-
                          realization noise statistics across all channels at once.
fiber/four_wave_mixing.py  Four-wave mixing (FWM) between discrete WDM channels: an analytical,
                          undepleted-pump treatment (like brillouin.py, a steady-state calculation
                          rather than a dynamical propagator -- true FWM needs the coherent
                          A_i*A_j*conj(A_k) field term, which WDMPropagator's intensity-only
                          XPM/Raman terms don't carry). fwm_efficiency() gives the standard phase-
                          matching factor (=1 exactly at zero mismatch, in both lossy and lossless
                          limits); fwm_power() the generated idler power for one pump triplet;
                          fwm_ghost_tone_power() sweeps every triplet from a given classical-channel
                          set and sums (incoherently) whatever lands within a tolerance of a target
                          wavelength -- e.g. a quantum channel's -- directly answering "how much FWM
                          ghost-tone power could land on my quantum channel". The degeneracy
                          prefactor (D=1 degenerate / D=2 non-degenerate) is a representative
                          convention, not independently verified against a reference -- treat
                          absolute FWM power as order-of-magnitude; the phase-matching efficiency
                          itself (which triplets matter, and how spacing/dispersion suppresses them)
                          is on firmer ground.
fiber/polarization.py    PolarizationPropagator: a 2-component (Jones vector) split-step GNLSE
                          solver -- everything else in fiber/ is scalar/implicitly single-polarized.
                          Adds polarization mode dispersion (PMD, via the standard "coarse-step"
                          method: random per-step axis rotation + fixed local differential group
                          delay, giving the correct RMS(DGD) ~ D_PMD*sqrt(L) random-walk scaling,
                          D_PMD in the usual ps/sqrt(km) telecom spec unit -- verified via Jones
                          Matrix Eigenanalysis on the composed operators, decoupled from any
                          particular pulse, to within statistical sampling error across a 16x range
                          in L) and polarization-dependent cross-phase modulation (2/3 for
                          orthogonal polarizations, a standard, confidently-established result from
                          chi^(3) tensor symmetry -- distinct from the representative/approximate
                          parameters elsewhere in this module) and Raman gain (a configurable,
                          approximate co/cross-polarization ratio). Reduces exactly to
                          FiberPropagator when launched purely into one component with no PMD.
fiber/brillouin.py       Stimulated Brillouin scattering (SBS): FiberMaterial gained g_B/nu_B/
                          delta_nu_B fields + brillouin_gain() Lorentzian spectrum; BrillouinPropagator
                          solves the steady-state coupled forward-pump/backward-Stokes power
                          equations as a two-point boundary value problem (shooting method), seeded
                          by a spontaneous-scattering noise floor; sbs_threshold_power() gives the
                          standard analytic (Smith 1972) threshold formula, over the shared kernel
                          sbs_threshold_from_aeff(A_eff, g_B, L_eff), which callers working from
                          computed modes (e.g. gmmnlse_raman_spectrum.py) use instead of repeating
                          21*A_eff/(g_B*L_eff) inline. A distinct mechanism
                          from Raman (acoustic vs optical phonons): ~10 GHz shift and ~tens-of-MHz
                          linewidth (vs ~13 THz / ~THz for Raman), predominantly backward-
                          scattering, with a far lower CW threshold power -- modelled as a
                          standalone steady-state power problem rather than on the fs-ps time grid
                          the other propagators use, since resolving the linewidth directly would
                          need a >30 ns simulation window
fiber/rayleigh_backscatter.py  Elastic Rayleigh backscattering: linear, non-stimulated, always-
                          present (no threshold, no frequency shift) -- the physics behind OTDR.
                          rayleigh_backscatter_power() gives the total backscattered power for a
                          CW/quasi-CW launch, derived directly from local-generation-rate times
                          double-pass attenuation, integrated over the fiber (saturates with length
                          rather than growing without bound); rayleigh_otdr_trace() gives the
                          classic time-resolved OTDR trace, self-consistent with the CW formula
                          (integrating it over return time reproduces the same total). alpha_R
                          (Rayleigh's share of total attenuation) and S (backscatter capture
                          fraction) are representative defaults, not measured values -- calibrate
                          against an OTDR measurement for precision; the formulas themselves
                          (saturation, double-pass attenuation, trace shape) are cross-checked
                          against brute-force numerical integration in
                          tests/test_rayleigh_backscatter.py.
fiber/hybrid_crosstalk.py  HybridCrosstalkPropagator: the first propagator in fiber/ combining
                          SPATIAL-mode diversity (like multimode_propagator.py) AND WAVELENGTH-
                          channel diversity (like wdm_propagator.py) at once -- a bright classical
                          reference in one mode group at one DWDM channel, one-way-coupled (its own
                          back-action from a weak/QKD-level peer field is negligible) into a weak
                          signal in a DIFFERENT mode group at a DIFFERENT channel. Combines two
                          already-validated pieces multiplicatively rather than deriving new
                          physics: the spatial part is MultimodeFiberParams.gamma_matrix[qkd,bright]
                          (the same coefficient multimode_propagator.py uses for intermodal XPM);
                          the spectral part is raman_gain_spectrum() evaluated at the fixed channel
                          separation (the same mechanism wdm_propagator.py uses for inter-channel
                          Raman crosstalk), called WITH gamma_matrix[qkd,bright] in place of a bare
                          gamma so the spatial overlap carries through. XPM is Kerr-only
                          (instantaneous, channel-separation-independent); deterministic Raman
                          crosstalk and spontaneous-Raman noise both use the bright field's local
                          instantaneous/peak power at the fixed spectral separation, matching
                          wdm_propagator.py's convention (not multimode_propagator.py's co-located
                          convolution, which assumes zero spectral offset). Also derives and applies
                          the ABSOLUTE phase of the bright field relative to the (reference, offset-0)
                          weak field -- combining MultimodeFiberParams.delta_beta0's intermodal term
                          with WDMPropagator's delta_beta0 construction (beta1_ref=material.n_g/c) --
                          the one piece of physics a pure power-domain (XPM/Raman) comparison cannot
                          give, and the reason this module exists: a phase-encoded protocol's
                          receiver measures exactly this relative phase.
fiber/receiver_leakage.py  filter_leakage_photons() (and its kernel leaked_photons_per_gate(), which
                          takes the received power directly and broadcasts, so studies computing
                          their own attenuation -- gmmnlse_coexistence.py -- share this formula):
                          a distinct, LINEAR, post-fiber mechanism from
                          everything else in fiber/ -- direct, un-shifted classical-carrier power
                          reaching the detector because a receive filter's real-world rejection
                          FLOOR (set by back-reflections, coating imperfections, secondary leakage
                          paths) is finite, not the (much steeper, but non-indefinite) roll-off
                          slope quoted near the passband edge. Converts a classical channel's
                          launch power, attenuated over the actual fiber length via that channel's
                          own per-mode fiber.alpha, into photons/gate at the QKD wavelength, given
                          an assumed or measured floor isolation. required_floor_dB() inverts this:
                          given a target noise budget, how much floor isolation is needed.
                          Independent of spatial mode by construction (this leakage happens in the
                          wavelength domain, downstream of the fiber) -- diagnostically useful when
                          spatial-mode diversity measurably fails to reduce classical-channel noise,
                          since that rules out mode-overlap-mediated mechanisms (Raman/XPM crosstalk)
                          as the dominant cause and points at this one instead.
fiber/constants.py       CODATA constants (via scipy): c, h, hbar, kB. The single source for every
                          module in fiber/ and studies/ -- each previously carried its own
                          hbar = 1.0545718e-34 (and materials.py its own kB), which is why this
                          module exists. c is exact; fiber/ previously imported c = 3e8 from
                          core/dfb_laser.py, a 0.07% error that shifts absolute propagation constants
                          by ~4e3 rad/m, comparable to the GRIN mode-group spacing. core/ still uses
                          its own c = 3e8 and h, so the laser models' published numbers are unchanged.
fiber/raman_models.py    Frequency-domain silica Raman responses: BlowWood (single oscillator) and
                          LinAgrawal (2006, adds the boson peak): peak g_R 5.8e-14 m/W at 1550 nm (vs
                          5.1e-14), and 1.4x / 2.2x more gain than Blow-Wood at 0.4 / 3 THz, where WDM
                          channel spacings sit. gain_shape / spontaneous_shape give the FDT weights.
fiber/grin_modes.py      Scalar LP-mode solver for alpha-profile GRIN and step-index fibres (finite-
                          volume radial eigenproblem, Malitson Sellmeier cladding, fixed Delta n):
                          beta_p(omega) fitted per mode over a span, guided-band masks, normalised
                          fields, the four-index overlap tensor S_plmn and intensity overlaps S_qlql.
                          field(p, x, y) samples a mode's normalised 2D transverse field (radial_profile
                          gives R(r), principal_group gives G = 2m + l - 1), which studies/mmf_mode_profiles.py
                          uses to render the LP profiles and check the l/m labelling.
                          Designs: om1, om3 (50/125, NA 0.2, alpha 2.05; shared by OM2-OM5) and smf28
                          (effective step NA 0.115, calibrated to Corning's A_eff, D, ZDW and cutoff).
                          OM3 at 1551.72 nm: 55 guided modes, A_eff(LP01) = 198 um^2, D = 22.1
                          ps/nm/km, LP11-LP01 walk-off 88 ps/km. Supersedes the reduced-order
                          parameters of multimode_fiber.py (A_eff 140 um^2, exponential overlap decay,
                          D = 17, calibrated DMD).
fiber/gmmnlse.py         GMMNLSE (Poletti & Horak 2008), RK4IP. Optional extensions, each off by
                          default so existing results are unchanged: polarisation=['x','y',...]
                          resolves the two polarisations of a spatial mode (exact 2/3 isotropic
                          Kerr factor cross-polarised, raman_copol_ratio on the Raman part, and
                          D_PMD for coarse-step PMD per mode); backward=True accumulates the
                          counter-propagating spontaneous PSD (no boundary-value iteration needed,
                          since spontaneous scattering does not deplete the pump -- stimulated
                          backward coupling stays in brillouin.py); vacuum_seed=True launches half
                          a photon per bin so the Kerr terms generate spontaneous FWM;
                          dispersive_overlaps=True lets the modes breathe with wavelength;
                          alpha_dB_km=ModeLoss(...) derives differential mode attenuation and
                          macrobend loss from the profiles; mode_coupling=ModeCoupling(...) adds
                          random linear coupling; and propagate(..., adaptive=tol) switches to
                          step-doubling error control with dz as the maximum step. Base solver: full per-mode
                          dispersion, loss, Kerr and Raman through S_plmn, self-steepening. A tensor
                          term is kept when phase matched to within coherence_tol (default 0.1/dz:
                          SPM, XPM, intermodal Raman gain, coherent coupling inside degenerate pairs;
                          coherence_tol = inf with small dz is the complete equation) and has at most
                          two spectator indices, a rule that preserves photon-number conservation.
                          Spontaneous Raman noise: noise='mean' integrates the ensemble-mean PSD in every
                          guided mode (pump spectrum convolved with the Raman kernel, no wrap-around,
                          at temperature=295 K by default, the laboratory value the studies use);
                          noise='stochastic' adds Langevin fields driven by local instantaneous pump
                          power. mode_coupling=ModeCoupling(kappa, correlation_length) adds random
                          LINEAR mode coupling (bends, splices, connectors) as one random unitary per
                          step, E|K_pq|^2 = kappa^2 w_pq dz with a Lorentzian weight
                          w = 1/(1 + (dbeta0 L_c)^2), so coupling favours quasi-degenerate modes,
                          conserves power exactly and accumulates diffusively (coupled power ~ kappa^2 w z).
                          Supersedes multimode_propagator.py, quantum_multimode.py and
                          hybrid_crosstalk.py for multimode work: those used intensity-only intermodal
                          coupling, which gives no Raman gain from a CW pump in another mode.
```

Validated in `tests/test_fiber_engine.py` against four independent physics checks: GVD-only Gaussian broadening vs the analytic formula, fundamental-soliton shape recurrence after one soliton period, Raman-induced soliton self-frequency shift (redshift), and spontaneous-Raman noise correctly vanishing on the anti-Stokes side as T->0. See `studies/raman_fiber_study.py` for a worked example (SSFS across fiber types; Stokes/anti-Stokes noise vs temperature).

`tests/test_multimode_fiber.py` validates the OM1-OM5 multimode support: monotonic OM1->OM5 bandwidth ordering matching nominal datasheet EMB/OFL figures, exact reduction to `FiberPropagator` when only one mode group is populated, OM1 broadening a multi-mode-launched pulse more than OM4 under pure intermodal dispersion (DMD), a broadband pump pulse in one mode group producing a measurable, Raman-specific (vanishing when decoupled) red-shifting pull on a probe pulse in a different mode group, per-mode loss (OM1 showing more differential mode attenuation than OM4, with the propagated power difference between mode 0 and the highest mode group matching the per-mode alpha difference to <0.05 dB), per-mode beta2/beta3 (nonzero spread across mode groups, with identical pulses launched into different mode groups broadening by measurably different amounts under pure GVD), and absolute inter-mode phase beta0 (an analytic self-consistency check -- differentiating delta_beta0 numerically reproduces delta_beta1's leading term to 1%-- plus an end-to-end propagation where the phase actually imprinted on a mode group's output matches the closed-form delta_beta0*L prediction to ~1e-10 rad). Note: intermodal coupling here only transfers *time-varying* power between mode groups -- a perfectly CW pump cannot seed frequency-selective gain in another mode group the way a same-mode two-tone pump/probe does in `FiberPropagator` (see the caveat in `fiber/multimode_propagator.py`'s docstring).

`tests/test_mode_coupling.py` validates `RandomModeCouplingPropagator`: exact power conservation across mode groups with no loss/nonlinearity (1e-15 relative error, expected since the coupling operator is exactly unitary by construction); `kappa=0` gives exactly zero coupling while `kappa>0` measurably couples power out of the launch mode; the coupled-away fraction grows monotonically with distance; and nearby mode groups pick up orders of magnitude more power than distant ones (locality from the overlap-decay weighting).

`tests/test_quantum_multimode.py` validates `QuantumMultimodePropagator`: mode 0's intramodal noise formula exactly matches `QuantumRamanPropagator`'s single-mode case; a quasi-CW signal in mode 0 with nothing launched elsewhere still seeds measurable spontaneous noise in other, initially-empty mode groups (unlike the classical deterministic intermodal term, which gives exactly zero for a CW driver), with the expected locality (adjacent mode >> distant mode); and the same Stokes/anti-Stokes asymmetry at T~0 already validated for the single-mode and WDM cases.

`tests/test_wdm_propagator.py` validates SPM/XPM/Raman-crosstalk/beta0: exact reduction to `FiberPropagator` for 1 channel; a weak co-propagating probe channel picks up exactly `gamma*2*P_pump*L` of XPM-induced phase (the standard XPM/SPM factor of 2) to within 0.3%; two CW-like channels ~13.2 THz apart show a genuine, sign-correct, resonance-selective Raman power transfer (which vanishes for closely-spaced channels or with Raman disabled) -- specifically demonstrating that unlike the multimode intermodal-Raman term, this one *does* work for unmodulated/CW channels, since it uses the fixed channel separation rather than a co-located baseband convolution; and absolute inter-channel phase beta0 (delta_beta0's derivative reproduces `beta1_ref + delta_beta1` to 1e-13 relative error -- these are exact closed-form polynomials here, not an approximation -- plus an end-to-end propagation matching the closed-form delta_beta0*L prediction exactly).

`tests/test_quantum_wdm.py` validates `QuantumWDMPropagator`: single-channel-limit consistency (intra-channel noise arrays match `QuantumRamanPropagator`'s exactly, inter-channel noise gain is identically zero with no peer channel), and the same Stokes/anti-Stokes physics as the single-mode quantum-noise test but for the inter-channel mechanism -- a strong channel spontaneously seeds noise in an initially-empty peer channel on the Raman gain side even at T~0, exactly zero on the loss side at T~0, and nonzero on the loss side once thermally activated (T=500K).

`tests/test_four_wave_mixing.py` validates FWM: efficiency eta=1 exactly at perfect phase matching (both lossy and lossless limits) and decreases with wider channel spacing; non-degenerate FWM generates exactly D^2=4x the degenerate power for otherwise identical parameters (a direct check of the formula's internal consistency); and a classic equally-spaced 3-channel comb produces a ghost tone landing exactly on the middle channel's own slot -- the textbook reason equal WDM channel spacing is avoided in real high-power systems -- correctly found and summed by `fwm_ghost_tone_power`.

`tests/test_polarization.py` validates `PolarizationPropagator`: exact reduction to `FiberPropagator` (0.00e+00 difference) when launched purely into one component with no PMD; exact power conservation under strong PMD with no loss/nonlinearity (2.7e-14 relative error); the cross-polarization XPM phase matching `gamma*(2/3)*P_pump*L` to within a few percent; and, via Jones Matrix Eigenanalysis on the composed per-step operators (isolating the underlying stochastic process from any particular pulse's response to it), RMS accumulated DGD matching the `D_PMD*sqrt(L)` scaling across a 16x range in length.

`tests/test_rayleigh_backscatter.py` validates elastic Rayleigh backscattering: the closed-form total-power formula matches brute-force numerical integration of the same underlying physics to 6e-6 relative error; backscattered power saturates to the analytic `alpha_R*S*P_in/(2*alpha)` asymptote as length grows rather than increasing without bound; and the time-resolved OTDR trace, integrated over return time, reproduces the same total power as the CW formula.

`tests/test_brillouin.py` validates the SBS solver against the textbook threshold picture: negligible SBS-specific pump depletion well below the analytic threshold, ~10% depletion right at threshold rising to >65% well above it (with reflectivity climbing from ~1e-8 to ~0.65 in between), reflectivity collapsing by 9 orders of magnitude when detuned 10 linewidths off resonance, and exact conservation (to 1e-15) of the net one-way photon flux `P_pump(z)-P_stokes(z)` for a lossless fiber. Two real bugs were caught and fixed while building this: a missing `1/A_eff` factor converting the standard tabulated (intensity-based) `g_B` into the power-based coupled equations, and a sign-convention mismatch feeding WDM channel separations into `raman_gain_spectrum` (whose `Omega` argument follows `physical_freq = omega0 - Omega`, opposite to the direct channel-offset convention).

`tests/test_hybrid_crosstalk.py` validates `HybridCrosstalkPropagator` against both propagators it combines: with `qkd_mode == bright_mode` (no real spatial separation), it reproduces `WDMPropagator`'s output exactly (<2e-9 relative error, including the absolute inter-channel/inter-mode phase term) and `QuantumWDMPropagator`'s ensemble-averaged noise floor statistically (ratio 0.97, within sampling error over 20 realizations); the deterministic Raman crosstalk coefficient vanishes exactly at zero channel separation (`raman_gain_spectrum`'s `Im(H_R)=0` at `Omega=0`); and `gamma_cross` matches `MultimodeFiberParams.gamma_matrix[qkd,bright]` directly. Three real bugs were caught and fixed while building this: a sign error in the channel-offset argument passed to `raman_gain_spectrum` (same class of bug as the one caught in `test_brillouin.py`), a completely missing absolute inter-field phase (`delta_beta0`) term -- the one piece of physics this module exists to add -- and a noise-generation bug where the cross-field term used peak power + frequency-domain shaping (correct for the *intramodal* term) instead of `QuantumWDMPropagator`'s local-instantaneous-power + time-domain white-noise recipe (correct for a *fixed-separation* term); all three were caught by comparing directly against the limiting-case propagators rather than by inspecting the formulas. `HybridCrosstalkPropagator` also gained an optional `launch_extinction_dB` parameter modeling a mode-selective launch device's (e.g. photonic lantern) finite mode extinction: a fraction of the nominal bright-mode launch power appears directly in the QKD's own mode at z=0, using the FULL same-mode gamma (not the weaker gamma_cross) since it now occupies that waveguide -- validated by two checks: a huge extinction value reproduces the no-leak baseline exactly, and (the more interesting identity) with `qkd_mode == bright_mode` and `launch_extinction_dB=0` (leak field an exact power-for-power copy of bright), the deterministic Raman-crosstalk GAIN FACTOR applied to the QKD field's magnitude is exactly SQUARED relative to a single-pathway run -- not doubled -- since Raman crosstalk is a purely multiplicative `exp(0.5*g_R*P*dz)` term per step and doubling the coefficient doubles the exponent.

`tests/test_receiver_leakage.py` validates `fiber.receiver_leakage`: a 10 dB floor step gives exactly 10x fewer leaked photons (log-linear by construction); leaked power attenuates exactly with the classical channel's own `fiber.alpha[bright_mode]` over length; and `required_floor_dB` round-trips exactly through `filter_leakage_photons`.

`tests/test_grin_modes.py` validates the mode solver: Malitson index at 1550 nm; eigenvalues of an exact infinite parabolic profile against the 2D harmonic-oscillator result (k²n₁² − β² = 2κG), with error 1.7e-5 at dr = 50 nm and second-order convergence; the parabolic LP01 effective area against 2π/κ (3.7e-6); SMF-28 against Corning's datasheet (single mode, A_eff 85.4 µm², D 16.6 ps/nm/km, still single-mode at 1270 nm); OM3 overlap symmetries (full index-permutation symmetry, the exact 1/3 LP11a/LP11b coherent ratio, azimuthal selection rules); the dispersion fit reproducing solver β₀ to 2e-4 rad/m; and the 2D field sampler (`field`, `radial_profile`, `principal_group`): unit power over a sampling box, LP11a ⟂ LP11b to 1e-9, exactly 2*l* azimuthal sign changes for *l* = 0–3, and the group number 2m+l−1.

`tests/test_gmmnlse.py` validates the GMMNLSE against independent limits: reduction to the split-step `FiberPropagator` for LP01 with β2/β3 and Blow-Wood Raman (1e-6 relative field difference); self-steepening peak delay 3γP₀z/ω₀ (24.67 vs 24.69 fs); exact decoupling of the circular LP11 combinations (|A±|² conserved to 1e-11, which fails if the coherent S_aabb term is wrong); CW intermodal Raman gain LP01 → LP11a against the analytic exponent (5 significant figures); the ensemble-mean spontaneous-Raman PSD against ħω g P₀ e^(−αL) L (6.6e-4) and the stochastic Langevin fields against the same (ratio 1.016 over 8 runs); photon-number conservation with the complete 21-term three-mode tensor, Raman and self-steepening (1.3e-8); and random linear mode coupling (exactly unitary, 5.7e-15 power drift; per-step transfer variance κ²w·dz within 1% of analytic over 5000 draws of the coupling operator; the Lorentzian Δβ₀ weight recovered exactly, 3e-6, by driving both mode pairs from the same RNG stream; and coupled power growing ×2.25 over doubled distance, consistent with the linear accumulation expected of a random walk). Note the propagation-level ensemble converges only as 1/√runs, since |a|² after a random walk is exponentially distributed — hence the per-step test rather than a large ensemble. The later extensions are validated the same way, each against a target independent of the implementation: **adaptive stepping** reaches 7.5e-09 against a 5000-step reference in 429 steps, where 50 fixed steps of the same maximum size give 2.9e-03; **ModeLoss** shows the cladding power fraction rising from 0.0000 (LP01) to 0.0072 (LP91b) with DMA tracking it, a 5 mm bend leaving LP01 at 0.300 dB/km while stripping the highest-order mode at 8848 dB/km, and propagated powers matching exp(−α_p L) mode by mode; **dispersive overlaps** reproduce independently solved A_eff to <1% over 1450–1750 nm and scale the Stokes-peak PSD by 0.9320 against a predicted 0.9320 (−6.8% at 13.2 THz); the **vacuum seed** carries exactly 0.5000 photons/bin and, with a CW pump (which generates *no* sidebands by SPM alone — measured as exactly zero), grows modulation-instability sidebands peaking at 0.664 THz against the analytic √(2γP/|β₂|)/2π = 0.693 THz; **backward noise** matches the analytic (1−e^{−2αL})/2α while the forward term turns over at 1/α, giving backward/forward = 1.00, 1.02, 1.16, 4.57 at 1, 5, 14, 50 km; and **polarisation** reduces to the scalar solver to machine precision when one component is populated, gives a cross-polarised XPM factor of 0.6667 against the exact 2/3, and keeps PMD unitary to 2.2e-16.

`tests/test_noise_calibration.py` checks the absolute spontaneous-Raman PSD of the single-mode `QuantumRamanPropagator` against the same analytic result (ratio 0.990). Its frequency-domain Langevin amplitudes (and those of `QuantumMultimodePropagator`, the intra-channel term of `QuantumWDMPropagator` and the self-noise term of `HybridCrosstalkPropagator`) were missing a factor √N, because `ifft` divides by N, which made that noise N times too weak (30–33 dB for the grids used). Results from `om3_raman_noise_sweep*.py` generated before the fix are affected; the time-domain inter-channel and cross-field noise terms were correct.

---

> Figure provenance for thesis sections 5.4 and 6.4 (which script draws which figure, at which
> line, and where the numbers are stored) is in [FIGURES.md](FIGURES.md).

## Scripts and Their Purposes

### Core Infrastructure

| File | Purpose |
|------|---------|
| `dfb_laser.py` | `DFBLaserParams` dataclass + `make_laser('dfb'\|'fp')` factory; supports DFB (HR rear facet) and Fabry-Perot (cleaved facets) cavities; derived quantities recomputed automatically |
| `sld_injection.py` | SLD (superluminescent diode) model: `SLDParams`, `InjectionParams`, steady-state solver, ASE-to-injection-field conversion |
| `million_pulse_comparison.py` | Numba-compiled Euler-Maruyama solver for 10^5-10^6 pulse statistics; extracts per-pulse peak phase, peak power, timing |

### Physics Studies (Recommendations #1-6)

| # | File | What It Studies | Output Directory |
|---|------|-----------------|-----------------|
| 1 | `multimode_analysis.py` | Multi-longitudinal-mode competition (5 modes) with shared carrier reservoir; mode partition noise, SMSR, k-factor | `images/multimode/` |
| 3 | `waveform_optimisation.py` | Differential evolution optimisation of modulation waveform for user-selected objectives (jitter, phase randomness, power, balanced) | `images/waveform_opt/` |
| 5 | `carrier_transport_analysis.py` | Carrier transport effects (SCH capture time tau_cap) on gain-switching dynamics; sweep over capture times | `images/carrier_transport/` |
| 6 | `fiber_propagation.py` | Split-step Fourier propagation through SMF-28; chirp compensation in anomalous dispersion; SLD impact on fiber effects (uses `fiber.propagator.FiberPropagator`, Kerr-only) | `images/fiber_propagation/` |
| - | `raman_fiber_study.py` | Nonlinear fiber Raman scattering via `fiber/`: classical soliton self-frequency shift across SMF-28/HNLF/PCF, and quantum spontaneous-Raman noise (Stokes/anti-Stokes asymmetry) vs temperature | `images/raman_fiber/` |
| - | `multimode_fiber_study.py` | Multimode (OM1-OM5) fiber via `fiber/multimode_*`: intermodal dispersion (DMD) broadening ordering across OM1-OM5, and intermodal Raman scattering (pump pulse in one mode group cross-Raman-shifting a probe pulse in another) | `images/multimode_fiber/` |
| - | `wdm_brillouin_study.py` | WDM effects via `fiber/wdm_propagator.py`: XPM-induced chirp on a probe channel, Raman tilt across a 9-channel comb; stimulated Brillouin scattering via `fiber/brillouin.py`: the classic reflectivity/transmission threshold knee vs input power | `images/wdm_brillouin/` |
| - | `om3_raman_noise_sweep.py` / `om3_raman_noise_sweep_pulsed.py` | Spontaneous Raman noise floor in OM3 (quasi-CW and 1 GHz/100 ps pulsed) across 12 lengths x 9 injected powers, via `fiber/quantum_multimode.py`. Superseded by `gmmnlse_raman_spectrum.py`: results saved before the √N noise fix are 30–33 dB low, the two sweeps use different grids and noise bands so their pulsed/CW ratio is not meaningful, and the pulsed run drives noise with peak power | `images/om3_raman_noise/`, `images/om3_raman_noise_pulsed/` |
| - | `hybrid_bb84_crosstalk.py` | Combined mode+wavelength crosstalk via `fiber/hybrid_crosstalk.py`: a bright reference (quasi-CW or 1 GHz pulsed) in OM3 mode group 1 / DWDM Ch 32 vs. a phase-encoded BB84 QKD signal in mode group 0 / Ch 34 -- spontaneous Raman noise landing in the QKD frame and XPM-induced differential phase between the signal's two time bins, swept over length and bright power. Superseded by `gmmnlse_coexistence.py`: its two channels (192.90/193.10 THz = Ch29/Ch31 on the bench convention, 200 GHz apart, μ = 0.5) are not the experiment's (Ch30→Ch34, 400 GHz, μ = 0.4), and its propagator uses the Blow-Wood Raman response, ~1.4x weaker than Lin-Agrawal at 0.4 THz detuning | `images/hybrid_bb84_crosstalk/` |
| - | `gmmnlse_raman_spectrum.py` | Chapter 4 Raman characterisation with `fiber/gmmnlse.py` on computed OM3 modes: CW 1551.72 nm pump in LP01 at 9 mW, ensemble-mean forward PSD in all 55 guided modes at 1–17 km (LP01-only vs total, per principal group), counter-propagating PSD from the solver's own `backward=True` accumulation (it replaced a duplicate analytic form in this study, which it reproduces to 2e-6), SMF-28 references (30 km/30 mW, 10 km/9 mW), mode table, and peak Stokes PSD vs spool length at fixed and SBS/launch-ceiling-limited power. Plotted by the Chapter 4 cells of `replot_paper_10ghz.ipynb` | `images/gmmnlse_raman_spectrum/` (data in `data/`) |
| - | `gmmnlse_coexistence.py` | Chapter 4 coexistence noise with `fiber/gmmnlse.py`, for two classical configurations in LP11b — the original −30 dBm Ch30 SFP (anti-Stokes side) and the current 2.6 mW CW laser at 1547 nm / 193.79 THz (Stokes side, +390 GHz from Ch34) — each with its 15 dB lantern leak into LP01: mean spontaneous-Raman photons per gate at the Ch34/LP01 QKD channel (0.05 nm filter, 500 ps gate) via overlap and via leak, Raman-only QBER, XPM phase on the QKD tone, and the receiver-filter-leakage QBER maps for the SFP case (`Sim_QBER_vs_FilterFloor`, `Sim_QBER_vs_Length_Floors`). Launch powers are taken as power in the fibre, so the laser case is an upper bound. Also sweeps distributed random linear mode coupling for the laser case (`ModeCoupling`, ensemble of 8 seeds), parametrised by the measurable LP11b→LP01 crosstalk accumulated at 10 km rather than by κ |
| - | `mmf_mode_profiles.py` | Mode profiles of OM3 at 1550 nm from `fiber/grin_modes.py`, as a check on the thesis LP-mode table: 2D fields (signed, showing the *l* phase cycles) and intensities (to sit beside a measured near field) of LP01/02/03, LP11/12/13, LP21/22/23, each verified to show *m* radial maxima and 2*l* azimuthal lobes; plus the principal-group structure G = 2m+l−1 against Gloge's β_G, mode counts per group, and the V-number scaling (V = 20.3, 55 spatial modes in 10 groups, 110 with polarisation). Plotted by the Chapter 3 cells of `replot_paper_10ghz.ipynb` | `images/mmf_mode_profiles/` (data in `data/`) |
| - | `paper_10ghz_simulation.py` | Numerical reproduction of the 10 GHz phase-randomised gain-switched DFB + SLD experiment (Lo et al., arXiv:2601.04031): stochastic rate equations with Langevin + ASE noise, r1/KL/KS randomisation metrics with confidence intervals, timing-jitter vs randomisation trade-off, filtered and unfiltered optical spectra, and an injection-power sweep | `images/paper_10ghz/` |
| - | `acceptance_bw_comparison.py` | The Part 1 frequency sweep repeated at three SLD acceptance bandwidths (0.6 nm ≈ the 77 GHz cavity resonance, 3.0 nm, 8.0 nm), comparing phase-randomisation quality, absolute jitter and AMZI outputs; writes per-bandwidth scalar JSON plus a comparison figure | `images/acceptance_bw_comparison_v2/` (data in `data/`; the un-suffixed `acceptance_bw_comparison/` holds an older run) |
| - | `gs_phase_sweep.py` | Phase correlation of a gain-switched DFB with CW SLD injection from 1–10 GHz: 10⁶ pulses per rate via vectorised Euler–Maruyama over 200 parallel realisations, giving P(S), the pulse-to-pulse intensity autocorrelation r(Δ) and the phase coherence \|g⁽¹⁾(Δ)\| | PDF report |
| - | `gs_injected_statistics.py` | Pulse-to-pulse statistics of ~10⁵ consecutive gain-switched pulses under CW SLD injection (Lang–Kobayashi terms, Langevin noise on carrier/photon/phase): sampled intensity distribution and intensity autocorrelation vs lag (`gs_inj_stats_*`) | `images/gain_switched_statistics/` | `images/gmmnlse_coexistence/` (data in `data/`) |

### PINN / Machine Learning Studies (Recommendations #7-12)

| # | File | Approach | Output Directory |
|---|------|----------|-----------------|
| 7 | `pinn_inverse_extraction.py` | Inverse PINN: extract (alpha_H, epsilon, beta_sp, a) from intensity-only time series | `images/pinn_inverse/` |
| 8 | `pinn_bandwidth.py` | HistogramNet: map AMZI visibility histograms to (bandwidth, coupling) | `images/pinn_bandwidth/` |
| 9 | `neural_surrogate.py` | MLP surrogate: (f_rep, S_inj, duty, I_bias) -> pulse statistics | `images/neural_surrogate/` |
| 10 | `neural_ode_noise.py` | Neural SDE: learn stochastic noise amplitudes via moment-matching | `images/neural_ode/` |
| 11 | `bayesian_pinn.py` | Bayesian PINN with MC Dropout for uncertainty quantification on extracted parameters | `images/bayesian_pinn/` |
| 12 | `transfer_learning.py` | Pre-train on simulation, fine-tune on sparse experimental data from a different device | `images/transfer_learning/` |

### Other Analysis Scripts

| File | Purpose | Output Directory |
|------|---------|-----------------|
| `phase_transition_analysis.py` | Maps the phase-coherence transition as a function of bias/modulation | `images/phase_transition/` |
| `qkd_source_analysis.py` | Full QKD source characterisation (QBER, key rate estimates) | `images/qkd_source/` |
| `qkd_sinj_sweep.py` | Sweep SLD injection power and measure r1 vs S_inj | `images/qkd_source/` |
| `timing_jitter_analysis.py` | Detailed timing jitter decomposition (turn-on delay statistics) | `images/timing_jitter/` |
| `detector_imperfections.py` | Impact of electronic noise and DC baseline offset on the single-photodiode g^(m)(0) estimator; derives and verifies the bias law (bias/excess = 10^-SNR/10), and tests the noise correction | `images/detector_imperfections/` |
| `amzi_pulse_analysis.py` | Simulated asymmetric Mach-Zehnder interferometer measurements | `images/amzi/` |
| `gain_switched_interference.py` | Pulse-to-pulse interference visibility (free-running) | `images/gain_switched/` |
| `gs_injected_interference.py` | Same with SLD injection | `images/gain_switched_injection/` |

---

## Key Parameters (Default Device)

### DFB Laser (default)
```
Wavelength:        1550 nm
Cavity length:     300 um
Active volume:     1.2e-17 m^3
Confinement:       Gamma = 0.3
Differential gain: a = 2.5e-20 m^2
Transparency:      N_tr = 1.5e24 m^-3
Gain compression:  epsilon = 3e-23 m^3
Alpha_H:           3.0
Beta_sp:           1e-4
R1 / R2:           0.32 / 0.95 (HR rear facet)
Mirror loss:       ~1985 m^-1
Threshold current: ~16.7 mA
Photon lifetime:   ~3.1 ps
Front facet frac:  0.93
```

### Fabry-Perot Laser
```
Same material/gain params as DFB, but:
R1 / R2:           0.32 / 0.32 (both cleaved facets)
Mirror loss:       ~3798 m^-1  (1.9x higher)
Threshold current: ~21.1 mA
Photon lifetime:   ~2.1 ps
Front facet frac:  0.50
No grating → all modes compete equally within gain bandwidth
```

Create via: `make_laser('fp')` or `make_laser('fp', L=500e-6)` for overrides.

---

## Usage

All scripts are run from the `Simulations/` root directory using `python3 -m`:

### Core Simulations

```bash
python3 -m core.million_pulse_comparison     # 1M-pulse phase correlation, 1-10 GHz
```

### Physics Studies

```bash
python3 -m studies.multimode_analysis        # Multi-mode competition (5 modes)
python3 -m studies.fiber_propagation         # SSFM fiber propagation
python3 -m studies.raman_fiber_study         # Raman SSFS + quantum noise vs temperature
python3 -m studies.multimode_fiber_study     # OM1-OM5 intermodal dispersion + Raman
python3 -m studies.wdm_brillouin_study       # WDM XPM/Raman crosstalk + SBS threshold
python3 -m studies.carrier_transport_analysis # Carrier transport effects

# Waveform optimisation (CLI with options)
python3 -m studies.waveform_optimisation --objective min_jitter --freq 5
python3 -m studies.waveform_optimisation --objective all --freq 5

python3 -m studies.phase_transition_analysis # Phase transition mapping
python3 -m studies.qkd_source_analysis       # QKD source characterisation
python3 -m studies.qkd_sinj_sweep            # S_inj threshold sweep
python3 -m studies.timing_jitter_analysis    # Jitter decomposition
python3 -m studies.amzi_pulse_analysis       # AMZI visibility analysis
```

### PINN / ML Experiments

```bash
python3 -m pinn.pinn_inverse_extraction      # Inverse parameter extraction
python3 -m pinn.pinn_bandwidth               # Histogram -> bandwidth
python3 -m pinn.neural_surrogate             # MLP surrogate model
python3 -m pinn.neural_ode_noise             # Neural SDE noise learning
python3 -m pinn.bayesian_pinn               # Bayesian PINN (MC Dropout)
python3 -m pinn.transfer_learning            # Transfer learning
```

### Tests

```bash
python3 -m tests.test_coherence_threshold
python3 -m tests.test_threshold_sweep
```

---

## Key Physical Findings

### 1. Multi-mode Competition (#1)
- Multi-mode competition **increases** phase coherence and **reduces** intensity noise compared to single-mode (counter-intuitive)
- SLD injection dramatically worsens mode partition noise (MPN k-factor increases from ~0.15 to ~0.35)
- SMSR degrades from >30 dB (free-running) to ~15 dB (SLD-injected)
- The dominant mode carries most of the useful signal; side modes add noise
- **DFB vs Fabry-Perot**: FP SMSR ~16 dB vs DFB ~34 dB (free-running at 5 GHz) due to absence of grating-based mode selectivity. FP mode partition noise is correspondingly worse.

### 2. Waveform Optimisation (#3)
- Raised-cosine is near-optimal at 5 GHz free-running
- Differential evolution finds ~5-10% improvements in specific objectives but with diminishing returns
- The optimisation landscape is relatively flat near the default operating point
- Fourier parameterisation (3 harmonics, 6 parameters) gives sufficient degrees of freedom

### 3. Detector Imperfections (Chapter 6)
- The bias of the uncorrected g^(m) estimator, as a fraction of the coherence excess, is exactly 10^(-SNR/10) for additive noise, independent of the source. The "13 dB" floor is therefore a 5% criterion.
- Verified numerically to 0.2% median deviation at m=2; m=3 and 4 track the same law within a few percent.
- A DC baseline offset of fraction delta inflates the excess by (1-delta)^-2, at every SNR. It does not diminish at high optical power and is NOT removed by the variance-based noise correction.
- The correction of Eqs. (6.13)-(6.14) removes 2-4 orders of magnitude of bias; the residual grows with order.
- Operating point bias=0.95*I_th, peak=1.6*I_th at 1 GHz reproduces the measured 19 mA source (g2/g3/g4 = 1.00057/1.00170/1.00337 against 1.0007/1.002/1.004).

### 4. Carrier Transport (#5)
- NOT the explanation for the experimental phase transition or discrepancies
- Main effect: ~10% power penalty and extra timing jitter at tau_cap > 5 ps
- Negligible effect on phase correlation at typical InGaAsP capture times (~1-3 ps)

### 5. Fiber Propagation (#6)
- Chirp compensation effect: pulses compress before broadening in anomalous dispersion regime
- Optimal compression distance: ~5-10 km for typical chirped pulses at 1550 nm
- SLD injection destroys chirp coherence, eliminating the compression benefit
- After 50+ km, GVD-dominated broadening regardless of initial chirp

### 6. PINN/ML Summary (#7-12)
- **Transfer learning works**: pre-training provides good initialisation for new devices
- **Neural surrogates** useful for "easy" statistics (power, width) but fail on phase metrics
- **Inverse PINNs fail** due to fundamental parameter identifiability issues from intensity-only data
- **Neural SDEs** remain an open research challenge (differentiating through stochastic paths)
- **Bayesian PINNs** give overconfident posteriors with poor coverage

---

## Laser Factory

```python
from core.dfb_laser import DFBLaserParams, make_laser

dfb = make_laser('dfb')                   # Default DFB (R2=0.95)
fp  = make_laser('fp')                    # Fabry-Perot (R1=R2=0.32)
fp2 = make_laser('fp', L=500e-6)          # FP with longer cavity
custom = DFBLaserParams(R2=0.70)          # Direct construction with overrides
```

The `cavity_type` field is carried through to `mode_setup()` in the multi-mode
analysis, which skips the DFB grating penalty for FP cavities.

---

## Shared Utilities (`gsdfb/` Package)

All analysis scripts import from the `gsdfb` package rather than duplicating common code.

### `gsdfb.analysis`
```python
from gsdfb import compute_r1, compute_metrics

r1, dphi = compute_r1(phi)               # Phase correlation + wrapped differences
metrics = compute_metrics(phi, pk_S, pk_k, dt, laser)  # Full QKD metric dict

from gsdfb.analysis import phase_randomisation_quality, absolute_jitter
pq = phase_randomisation_quality(phi)     # r1, KL divergence, KS stat
sigma_t, t_peak = absolute_jitter(pk_k, dt)
```

### `gsdfb.plotting`
```python
from gsdfb import setup_plotting, save_fig
from gsdfb.plotting import img_dir

setup_plotting()                          # Apply consistent matplotlib style
save_fig(fig, 'images/myplot/fig1.png')   # Save + makedirs + print path
out = img_dir('myplot')                   # Create and return 'images/myplot'
```

### `gsdfb.pinn_utils`
```python
from gsdfb.pinn_utils import (
    Scales, LaserPINN, LearnableParams,
    physics_residuals, compute_loss,
    generate_reference, generate_gain_switched_data,
    add_noise_snr, make_collocation, subsample,
    to_tensor, DEVICE, DTYPE,
)
```

---

## Dependencies

```
numpy
scipy
matplotlib
numba
torch (for PINN/ML scripts only)
```

Install:
```bash
pip install numpy scipy matplotlib numba torch
```

---

## File Organisation

```
Simulations/
│
├── core/                               # Core simulation engine
│   ├── dfb_laser.py                    #   DFBLaserParams dataclass + ODE solvers
│   ├── sld_injection.py                #   SLD model, injection coupling, Lang-Kobayashi
│   ├── million_pulse_comparison.py     #   Numba JIT stochastic solver + waveform builders
│   └── gain_switched_interference.py   #   GainSwitchParams, autocorrelation, MZI functions
│
├── gsdfb/                              # Shared utilities package
│   ├── analysis.py                     #   Phase metrics (r1, KL, KS), jitter, AMZI
│   ├── plotting.py                     #   Matplotlib config, save_fig, img_dir
│   └── pinn_utils.py                   #   Scales, LaserPINN, LearnableParams,
│                                       #   physics_residuals, compute_loss, add_noise_snr
│
├── studies/                            # Physics analysis scripts
│   ├── multimode_analysis.py           #   Multi-mode competition (#1)
│   ├── waveform_optimisation.py        #   Waveform optimisation CLI (#3)
│   ├── carrier_transport_analysis.py   #   Carrier transport (#5)
│   ├── fiber_propagation.py            #   Fiber propagation SSFM (#6)
│   ├── phase_transition_analysis.py    #   Phase transition mapping
│   ├── qkd_source_analysis.py          #   QKD source characterisation
│   ├── qkd_sinj_sweep.py              #   S_inj threshold sweep
│   ├── timing_jitter_analysis.py       #   Jitter decomposition
│   ├── amzi_pulse_analysis.py          #   AMZI simulation
│   ├── gs_injected_interference.py     #   Injected interference
│   ├── gs_injected_statistics.py       #   Injection pulse statistics
│   └── gs_phase_sweep.py              #   Phase sweep (PDF report)
│
├── pinn/                               # PINN / ML experiment scripts
│   ├── pinn_inverse_extraction.py      #   Inverse PINN (#7)
│   ├── pinn_bandwidth.py               #   Histogram -> bandwidth (#8)
│   ├── neural_surrogate.py             #   MLP surrogate (#9)
│   ├── neural_ode_noise.py             #   Neural SDE (#10)
│   ├── bayesian_pinn.py                #   Bayesian PINN (#11)
│   └── transfer_learning.py            #   Transfer learning (#12)
│
├── tests/                              # Validation tests
│   ├── test_coherence_threshold.py     #   Coherence destruction threshold
│   └── test_threshold_sweep.py         #   eta_coupling sweep
│
├── archive/                            # Superseded (not imported)
│   ├── laser_pinn_monolithic.py
│   └── sld_standalone_1300nm.py
│
├── images/                             # All output figures (auto-created)
│
└── DOCUMENTATION.md
```

---

## Further Steps

### High-Priority (Directly Actionable)

1. **Experimental validation of chirp compensation**
   - The fiber propagation simulation predicts a compression sweet-spot at ~5-10 km. This is measurable with a streak camera or fast photodiode + sampling scope. Compare pulse widths at 0, 5, 10, 20 km fiber lengths.

2. **Phase-resolved measurements for parameter extraction**
   - The PINN inverse extraction failed because intensity-only data leaves alpha_H and epsilon degenerate. A heterodyne or self-homodyne measurement of the optical field would break this degeneracy. Consider adding a coherent detection arm to the existing AMZI setup.

3. **Multi-mode rate equation with measured gain spectrum**
   - The 5-mode model uses a parabolic gain approximation. Replace with a measured/fitted ASE spectrum shape from the actual device. This will give quantitative MPN predictions rather than qualitative trends.

4. **Transfer learning on real experimental data**
   - The framework in `transfer_learning.py` is ready. Pre-train on simulation, then fine-tune on actual experimental time traces. The key question: does the physics prior in the PINN regularisation actually help when the model is slightly wrong?

5. **Waveform optimisation at higher repetition rates**
   - The current results are at 5 GHz where the laser is well-behaved. At 10+ GHz the dynamics become much more constrained and waveform shaping may yield larger improvements. Run:
   ```bash
   python3 waveform_optimisation.py --objective balanced --freq 10 --n_harmonics 4
   ```

### Medium-Priority (Require More Development)

6. **Incorporate temperature dependence**
   - Add T-dependent gain parameters: a(T), N_tr(T), and thermal roll-off. The current model assumes isothermal operation but real devices heat up during high-duty-cycle operation, shifting threshold and gain.

7. **Correlated noise model for multi-mode**
   - The current multi-mode model uses independent Langevin sources per mode. In reality, carrier noise couples all modes through the shared reservoir. Implement a correlated noise matrix with cross-spectral terms.

8. **Full QKD security analysis pipeline**
   - Connect the pulse statistics (r1, intensity distributions) to an actual QKD security proof. Compute the mutual information leakage from residual coherence and translate to key rate penalty. Compare with the decoy-state BB84 bounds.

9. **GPU-accelerated solver for parameter sweeps**
   - The current Numba solver is CPU-bound. For large parameter sweeps (e.g., 2D maps of r1 vs bias and frequency), port the solver to CuPy or write a custom CUDA kernel. Expected speedup: 10-50x for batch simulations.

10. **Dispersion-managed link optimisation**
    - Extend `fiber_propagation.py` to model DCF (dispersion-compensating fiber) spans. Optimise the DCF length and placement for minimum pulse distortion at the receiver while maintaining phase randomness.

### Lower-Priority (Exploratory)

11. **Reservoir computing with gain-switched dynamics**
    - The nonlinear transient dynamics of the gain-switched laser could serve as a physical reservoir computer. Train a linear readout on the transient ring-down features to classify input waveforms.

12. **Stochastic optimal control formulation**
    - Reformulate waveform optimisation as a proper stochastic optimal control problem (Pontryagin's principle with the rate equations as constraints). This would give a continuous-time optimal drive rather than a discrete parameterisation.

13. **Polarisation-resolved model**
    - Add TE/TM mode splitting for polarisation-multiplexed QKD. The current scalar model ignores polarisation-dependent gain and birefringence.

14. **Feedback effects (external cavity)**
    - Model the effect of residual back-reflections from fiber connectors. Even -40 dB feedback can significantly perturb phase dynamics in gain-switched lasers.

15. **Score-matching for Neural SDE training**
    - The moment-matching loss in `neural_ode_noise.py` failed because gradients through stochastic paths are noisy. Try Stein score matching or denoising score matching as alternative loss functions that avoid differentiating through the SDE sample paths.
