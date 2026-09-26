"""Thesis figure for Chapter 7: SLD injection into a Fabry-Perot and a DFB laser at 10 GHz.

Reads images/fp_randomisation/diag_fixed_inj10g.csv, written by the fixed-delay diagnostic run
described in studies/fp_session_report.md (300 um FP, I_off = 0.9 I_th, 12000 pulses per point,
fixed-delay sampling). No simulation is run here.

    python3 studies/fp_thesis_figure.py

Output: images/fp_randomisation/fp_injection_10ghz.png (+ manifest).
"""
import csv
import sys

import numpy as np

sys.path.insert(0, '.')
from gsdfb.plotting import setup_plotting, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CSV_PATH = 'images/fp_randomisation/diag_fixed_inj10g.csv'
OUT_PATH = 'images/fp_randomisation/fp_injection_10ghz.png'
R1_CRITERION = 0.01          # security criterion used in Chapter 5
N_PULSES = 12000
R1_FLOOR = 0.886 / np.sqrt(N_PULSES)   # |r1| of truly random phases for N pulses
LINTHRESH = 1e18             # symlog: linear below, so the free-running point sits at 0


def load(model):
    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r['model'] == model]
    col = lambda k: np.array([float(r[k]) for r in rows])
    return col('total'), col('r1w_fixed'), col('r1_line_fixed'), col('dfb')


def main():
    setup_plotting()
    total, fp_all, fp_line, dfb = load('original')
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhspan(1e-4, R1_FLOOR, color='0.9', lw=0, label=f'floor, $N={N_PULSES}$')
    ax.axhline(R1_CRITERION, color='0.3', ls=':', lw=1, label=r'$r_1 = 0.01$')
    ax.plot(total, dfb, 's-', color='tab:blue', label='DFB')
    ax.plot(total, fp_all, 'o-', color='tab:orange', label='FP, all lines')
    ax.plot(total, fp_line, 'o--', color='tab:orange', mfc='white', label='FP, strongest line')
    ax.set_xscale('symlog', linthresh=LINTHRESH)
    ax.set_yscale('log')
    ax.set_xlim(-2e17, 2e21)
    ax.set_ylim(3e-3, 1.5)
    ax.set_xlabel(r'Total injected photon density (m$^{-3}$)')
    ax.set_ylabel(r'Lag-one phase correlation $r_1$')
    ax.set_xticks([0, 1e18, 1e19, 1e20, 1e21])
    ax.set_xticklabels(['0', r'$10^{18}$', r'$10^{19}$', r'$10^{20}$', r'$10^{21}$'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc='upper right', frameon=False)
    save_fig(fig, OUT_PATH, params={'csv': CSV_PATH, 'model': 'original', 'f_rep_GHz': 10,
                                    'I_off_rel': 0.9, 'cavity_um': 300, 'n_pulses': N_PULSES,
                                    'sampling': 'fixed delay'})
    print('wrote', OUT_PATH)


if __name__ == '__main__':
    main()
