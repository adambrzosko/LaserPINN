"""
Mode profiles supported by OM3 at 1550 nm, computed with fiber/grin_modes.py, as a direct
check on the LP-mode table in the multimode-fibre chapter and the claims made around it:

  * the nine LP_lm modes listed in the table (LP01/02/03, LP11/12/13, LP21/22/23) are all
    guided in OM3 at 1550 nm, and each shows the structure its label asserts -- m intensity
    maxima along the radius and 2l lobes around the azimuth (l phase cycles);
  * modes with l >= 1 come as an orientation pair (a = cos l.phi, b = sin l.phi), which is
    what makes them four-fold degenerate once the two polarisations are counted, against
    two-fold for l = 0;
  * the guided modes organise into principal mode groups G = 2m + l - 1 of nearly equal
    propagation constant, following Gloge's beta_G = n1 k0 sqrt(1 - 2 Delta G/P);
  * OM3 at 1550 nm supports ten such groups, i.e. P(P+1)/2 = 55 spatial modes (110 with
    polarisation), consistent with V = 20.3 and the M ~ V^2/4 scaling.

Writes images/mmf_mode_profiles/{mmf_lp_mode_profiles.png (signed field, showing the l
phase cycles), mmf_lp_mode_intensity.png (intensity, to sit beside a measured near-field
image), mmf_mode_groups.png} and data/{parameters.json, profiles.npz, modes.csv}.

    python studies/mmf_mode_profiles.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from gsdfb import save_fig
import numpy as np

from fiber.constants import c
from fiber.grin_modes import DESIGNS, FibreModes, silica_index

OUT = Path(__file__).resolve().parent.parent / 'images' / 'mmf_mode_profiles'
DATA = OUT / 'data'

WAVELENGTH = 1550e-9
DESIGN = DESIGNS['om3']
SPAN_HZ = 5e12
TABLE_MODES = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
BOX_UM = 40.0
N_GRID = 401


def label_of(l, m):
    return f'LP{l}{m}' + ('' if l == 0 else 'a')


def radial_maxima(r, R, core_radius):
    """Number of intensity maxima along the radius, counted inside the core."""
    I = R ** 2
    inner = r < core_radius
    peak = I[inner].max()
    interior = (I[1:-1] > I[:-2]) & (I[1:-1] >= I[2:]) & (I[1:-1] > 0.01 * peak)
    return int(np.count_nonzero(interior[inner[1:-1]])) + int(I[0] > I[1] and I[0] > 0.01 * peak)


def azimuthal_lobes(modes, p, r_peak):
    """Number of sign changes of the field around a ring at r_peak (= 2l for l >= 1)."""
    phi = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    ring = modes.field(p, r_peak * np.cos(phi), r_peak * np.sin(phi))
    return int(np.count_nonzero(np.diff(np.sign(ring), append=np.sign(ring[0]))))


def run():
    DATA.mkdir(parents=True, exist_ok=True)
    modes = FibreModes(DESIGN, WAVELENGTH, span_Hz=SPAN_HZ)

    k0 = 2 * np.pi / WAVELENGTH
    n1 = silica_index(WAVELENGTH) + DESIGN.dn0
    V = k0 * DESIGN.core_radius * DESIGN.NA
    Delta = (n1 ** 2 - silica_index(WAVELENGTH) ** 2) / (2 * n1 ** 2)

    groups = {}
    for p in range(len(modes)):
        groups.setdefault(modes.principal_group(p), []).append(p)
    P = max(groups)

    # ---- the nine modes of the table -------------------------------------------------
    g = np.linspace(-BOX_UM, BOX_UM, N_GRID) * 1e-6
    X, Y = np.meshgrid(g, g)
    fields, rows = [], []
    for l, m in TABLE_MODES:
        p = modes.index_of(label_of(l, m))
        r, R = modes.radial_profile(p)
        r_peak = float(r[np.argmax(R ** 2)])
        psi = modes.field(p, X, Y)
        fields.append(psi)
        rows.append({
            'label': f'LP{l}{m}', 'l': l, 'm': m, 'group': modes.principal_group(p),
            'beta_rad_m': float(modes.beta0[p]), 'A_eff_um2': modes.effective_area(p) * 1e12,
            'radial_maxima': radial_maxima(r, R, DESIGN.core_radius),
            'azimuthal_lobes': azimuthal_lobes(modes, p, r_peak),
            'orientations': 1 if l == 0 else 2,
            'degeneracy_with_polarisation': 2 if l == 0 else 4,
            'power_in_core': float(np.sum(R[r < DESIGN.core_radius] ** 2
                                          * r[r < DESIGN.core_radius]) * modes.dr),
        })
    fields = np.asarray(fields)

    print(f'OM3 at {WAVELENGTH * 1e9:.0f} nm: V = {V:.2f}, Delta = {Delta:.4f}, '
          f'{len(modes)} spatial modes in {len(groups)} principal groups')
    print(f'{"mode":6s} {"group":>5s} {"radial max":>10s} {"azim lobes":>10s} {"expected":>12s} '
          f'{"A_eff(um2)":>10s} {"core power":>10s}')
    for row in rows:
        expected = f'm={row["m"]}, 2l={2 * row["l"]}'
        print(f'{row["label"]:6s} {row["group"]:5d} {row["radial_maxima"]:10d} '
              f'{row["azimuthal_lobes"]:10d} {expected:>12s} {row["A_eff_um2"]:10.1f} '
              f'{row["power_in_core"]:10.3f}')

    ok_radial = all(row['radial_maxima'] == row['m'] for row in rows)
    ok_azimuth = all(row['azimuthal_lobes'] == (2 * row['l'] if row['l'] else 0) for row in rows)
    print(f'radial index m reproduced: {ok_radial}; azimuthal index l reproduced: {ok_azimuth}')

    # ---- group structure -------------------------------------------------------------
    G = np.array(sorted(groups))
    counts = np.array([len(groups[q]) for q in G])
    beta_mean = np.array([modes.beta0[groups[q]].mean() for q in G])
    beta_spread = np.array([np.ptp(modes.beta0[groups[q]]) for q in G])
    gloge = n1 * k0 * np.sqrt(1 - 2 * Delta * (G / P))
    print(f'modes per group: {[int(x) for x in counts]} (sum {counts.sum()}; '
          f'P(P+1)/2 = {P * (P + 1) // 2})')
    print(f'V^2/4 = {V ** 2 / 4:.0f} modes with polarisation ({2 * len(modes)} computed), '
          f'V^2/8 = {V ** 2 / 8:.0f} per polarisation ({len(modes)} computed)')
    step = np.abs(np.diff(beta_mean))
    print(f'intra-group beta spread: {beta_spread.max():.1f} rad/m worst case, vs '
          f'{step.min():.1f}-{step.max():.1f} rad/m between neighbouring groups')

    # ---- figure 1: the table's nine modes ---------------------------------------------
    fig1, axes1 = plt.subplots(3, 3, figsize=(10.5, 10.5))
    for ax, psi, row in zip(axes1.ravel(), fields, rows):
        lim = np.abs(psi).max()
        ax.imshow(psi, extent=[-BOX_UM, BOX_UM, -BOX_UM, BOX_UM], cmap='RdBu_r',
                  vmin=-lim, vmax=lim, origin='lower')
        ax.add_patch(plt.Circle((0, 0), DESIGN.core_radius * 1e6, fill=False, lw=1.0,
                                color='k', alpha=0.5))
        ax.set_title(f'LP$_{{{row["l"]}{row["m"]}}}$  (group {row["group"]}, '
                     f'{row["degeneracy_with_polarisation"]}-fold)', fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    fig1.suptitle(f'Scalar LP mode fields of OM3 at {WAVELENGTH * 1e9:.0f} nm '
                  f'(core outlined, {2 * BOX_UM:.0f} '
                  r'$\mu$m box; red/blue = field sign)', fontsize=13)
    fig1.tight_layout()
    save_fig(fig1, OUT / 'mmf_lp_mode_profiles.png', dpi=150, bbox_inches='tight')

    # ---- figure 1b: the same modes as intensity, to sit beside a measured near field ---
    fig1b, axes1b = plt.subplots(3, 3, figsize=(10.5, 10.5))
    for ax, psi, row in zip(axes1b.ravel(), fields, rows):
        ax.imshow(psi ** 2, extent=[-BOX_UM, BOX_UM, -BOX_UM, BOX_UM], cmap='inferno',
                  origin='lower')
        ax.add_patch(plt.Circle((0, 0), DESIGN.core_radius * 1e6, fill=False, lw=1.0,
                                color='w', alpha=0.5))
        ax.set_title(f'LP$_{{{row["l"]}{row["m"]}}}$  ({row["m"]} radial, '
                     f'{2 * row["l"] if row["l"] else 1} azimuthal)', fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    fig1b.suptitle(f'LP mode intensities of OM3 at {WAVELENGTH * 1e9:.0f} nm '
                   f'(core outlined, {2 * BOX_UM:.0f} ' r'$\mu$m box)', fontsize=13)
    fig1b.tight_layout()
    save_fig(fig1b, OUT / 'mmf_lp_mode_intensity.png', dpi=150, bbox_inches='tight')

    # ---- figure 2: principal mode groups ----------------------------------------------
    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5))
    for q in G:
        axa.plot(np.full(len(groups[q]), q), modes.beta0[groups[q]], 'o', ms=7,
                 color='#1f77b4', alpha=0.7)
    axa.plot(G, gloge, 'k--', lw=1.5, label=r'Gloge $\beta_G = n_1 k_0\sqrt{1-2\Delta G/P}$')
    axa.plot([], [], 'o', color='#1f77b4', label='computed modes')
    axa.set_xlabel('Principal mode group $G = 2m + l - 1$')
    axa.set_ylabel(r'$\beta$ (rad/m)')
    axa.legend(fontsize=10)

    axb.bar(G, counts, color='#1f77b4', alpha=0.8, label='computed')
    axb.plot(G, G, 'k--', lw=1.5, label='$G$ modes in group $G$')
    axb.set_xlabel('Principal mode group $G$')
    axb.set_ylabel('Spatial modes in group')
    axb.set_title(f'{len(modes)} spatial modes, {len(groups)} groups '
                  f'({2 * len(modes)} with polarisation)', fontsize=12)
    axb.legend(fontsize=10)
    fig2.tight_layout()
    save_fig(fig2, OUT / 'mmf_mode_groups.png', dpi=150, bbox_inches='tight')

    # ---- data -------------------------------------------------------------------------
    np.savez(DATA / 'profiles.npz', x_um=g * 1e6, y_um=g * 1e6, fields=fields,
             labels=np.array([row['label'] for row in rows]),
             table_groups=np.array([row['group'] for row in rows]),
             table_l=np.array([row['l'] for row in rows]),
             table_m=np.array([row['m'] for row in rows]),
             table_degeneracy=np.array([row['degeneracy_with_polarisation'] for row in rows]),
             group_numbers=G, modes_per_group=counts, beta_mean=beta_mean,
             beta_spread=beta_spread, beta_gloge=gloge,
             all_beta=modes.beta0, all_labels=np.array(modes.labels),
             all_groups=np.array([modes.principal_group(p) for p in range(len(modes))]),
             core_radius_um=DESIGN.core_radius * 1e6)
    with open(DATA / 'modes.csv', 'w') as fh:
        fh.write('index,label,l,m,orientation,group,beta_rad_m,cutoff_nm\n')
        for p in range(len(modes)):
            mode = modes.modes[p]
            cutoff = modes.cutoff_wavelength(p)
            fh.write(f'{p},{mode.label},{mode.l},{mode.m},{mode.orientation or "-"},'
                     f'{modes.principal_group(p)},{modes.beta0[p]:.4f},'
                     f'{"" if np.isnan(cutoff) else f"{cutoff * 1e9:.1f}"}\n')
    (DATA / 'parameters.json').write_text(json.dumps({
        'design': DESIGN.name, 'wavelength_nm': WAVELENGTH * 1e9,
        'core_radius_um': DESIGN.core_radius * 1e6, 'NA': DESIGN.NA,
        'alpha_profile': DESIGN.alpha_profile, 'n1': float(n1), 'Delta': float(Delta),
        'V': float(V), 'n_spatial_modes': len(modes), 'n_with_polarisation': 2 * len(modes),
        'n_groups': len(groups), 'V2_over_4': float(V ** 2 / 4), 'V2_over_8': float(V ** 2 / 8),
        'P_from_V_over_2': float(V / 2), 'triangular_count': P * (P + 1) // 2,
        'modes_per_group': [int(x) for x in counts],
        'max_intra_group_beta_spread_rad_m': float(beta_spread.max()),
        'inter_group_beta_step_rad_m': [float(step.min()), float(step.max())],
        'table_modes': rows, 'radial_index_verified': ok_radial,
        'azimuthal_index_verified': ok_azimuth,
    }, indent=2))
    print(f'saved to {OUT}')


if __name__ == '__main__':
    run()
