"""
PINN Bandwidth Inference from AMZI Data (#8)

Inverse problem: given a histogram of AMZI splitting ratios eta,
infer the acceptance_bandwidth and eta_coupling parameters.

Approach:
  1. Use the numba solver simulate_pulses to generate pulse trains at 5 GHz
  2. Compute per-pulse phase differences Delta_phi between consecutive pulses
  3. Convert to AMZI splitting ratio: eta = cos^2(Delta_phi / 2)
  4. Build a neural network that maps histogram shape -> (bandwidth, coupling)
  5. Train on synthetic histograms spanning parameter space
  6. Test on held-out parameter combinations

Figures saved to images/pinn_bandwidth/:
  1. training_loss.png — loss vs epoch
  2. predictions.png — predicted vs true bandwidth/coupling for test set
  3. example_histograms.png — eta histograms at different bandwidths

Runtime target: < 10 minutes total.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numba import njit

from dfb_laser import DFBLaserParams, q, h, c
from million_pulse_comparison import simulate_pulses
from sld_injection import (
    SLDParams, InjectionParams,
    solve_sld_steady_state, sld_to_injection_field,
)

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'images', 'pinn_bandwidth')
os.makedirs(OUT_DIR, exist_ok=True)

DTYPE = torch.float32  # float32 is fine for this regression task


# ── Compute S_inj for given (acceptance_bandwidth, eta_coupling) ────────────

def compute_S_inj(laser, sld, P_sld, acceptance_bandwidth, eta_coupling):
    """
    Compute injected photon density for given bandwidth and coupling.

    Uses the SLD model to determine how much broadband power couples
    into the laser mode.
    """
    inj = InjectionParams(eta_coupling=eta_coupling)
    inj.compute_derived(laser)
    S_inj, _ = sld_to_injection_field(sld, P_sld, laser, inj,
                                       acceptance_bandwidth=acceptance_bandwidth)
    return S_inj


# ── Generate eta histogram for given parameters ─────────────────────────────

def generate_eta_histogram(laser, S_inj, n_pulses=10000, n_discard=500,
                           f_rep=5e9, n_bins=50, seed=42):
    """
    Simulate gain-switched pulses and compute AMZI splitting ratio histogram.

    Returns histogram bin counts (normalized) and bin edges.
    """
    I_th = laser.threshold_current()
    I_off = 0.9 * I_th
    I_on = 5.0 * I_th

    T_rep = 1.0 / f_rep
    t_on = 0.30 * T_rep
    DT = 1.0e-12
    pts_period = max(int(round(T_rep / DT)), 50)
    dt_actual = T_rep / pts_period
    t_rise = min(20e-12, t_on / 4.0)

    # Run numba solver
    peak_phi, peak_S, samp_S, peak_k = simulate_pulses(
        n_pulses + n_discard, n_discard, pts_period, dt_actual,
        I_off, I_on, t_on, t_rise,
        laser.V, laser.Gamma, laser.v_g, laser.a,
        laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C,
        laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        S_inj, seed,
    )

    # Phase differences between consecutive pulses
    dphi = np.diff(peak_phi)
    # Wrap to [-pi, pi]
    dphi = np.angle(np.exp(1j * dphi))

    # AMZI splitting ratio: eta = cos^2(dphi / 2)
    eta = np.cos(dphi / 2.0) ** 2

    # Histogram (normalized to sum to 1)
    counts, bin_edges = np.histogram(eta, bins=n_bins, range=(0.0, 1.0))
    counts_norm = counts.astype(np.float64) / counts.sum()

    return counts_norm, bin_edges, eta


# ── Generate training dataset ───────────────────────────────────────────────

def generate_training_data(laser, sld, P_sld, n_bw=10, n_eta=8,
                           n_pulses_train=15000, n_bins=50):
    """
    Generate synthetic eta histograms over a grid of (bandwidth, eta_coupling).

    Returns X (histograms), Y (log10 bandwidth, log10 eta_coupling).
    """
    # Parameter ranges
    bw_range = np.logspace(np.log10(100e9), np.log10(5e12), n_bw)   # 100 GHz to 5 THz
    eta_range = np.logspace(np.log10(0.01), np.log10(1.0), n_eta)   # 0.01 to 1.0

    X_list = []
    Y_list = []
    seed_counter = 0

    total = n_bw * n_eta
    print(f"    Generating {total} histograms ({n_bw} bandwidths x {n_eta} couplings)...")
    print(f"    Pulses per histogram: {n_pulses_train}")

    t0 = time.time()
    for i, bw in enumerate(bw_range):
        for j, eta_c in enumerate(eta_range):
            S_inj = compute_S_inj(laser, sld, P_sld, bw, eta_c)

            counts_norm, _, _ = generate_eta_histogram(
                laser, S_inj, n_pulses=n_pulses_train,
                n_discard=200, n_bins=n_bins, seed=seed_counter,
            )
            seed_counter += 1

            X_list.append(counts_norm)
            Y_list.append([np.log10(bw), np.log10(eta_c)])

        elapsed = time.time() - t0
        rate = (i + 1) * n_eta / elapsed
        remaining = (total - (i + 1) * n_eta) / max(rate, 1e-6)
        print(f"      bw={bw:.1e} Hz done ({(i+1)*n_eta}/{total}, "
              f"~{remaining:.0f}s remaining)")

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    return X, Y, bw_range, eta_range


def generate_test_data(laser, sld, P_sld, n_test=15,
                       n_pulses_test=30000, n_bins=50):
    """
    Generate held-out test set at random (bandwidth, eta_coupling) combinations.
    """
    rng = np.random.default_rng(999)
    # Random log-uniform samples
    log_bw = rng.uniform(np.log10(100e9), np.log10(5e12), n_test)
    log_eta = rng.uniform(np.log10(0.01), np.log10(1.0), n_test)

    X_list = []
    Y_list = []

    print(f"    Generating {n_test} test histograms ({n_pulses_test} pulses each)...")
    t0 = time.time()

    for i in range(n_test):
        bw = 10.0 ** log_bw[i]
        eta_c = 10.0 ** log_eta[i]
        S_inj = compute_S_inj(laser, sld, P_sld, bw, eta_c)

        counts_norm, _, _ = generate_eta_histogram(
            laser, S_inj, n_pulses=n_pulses_test,
            n_discard=300, n_bins=n_bins, seed=5000 + i,
        )

        X_list.append(counts_norm)
        Y_list.append([np.log10(bw), np.log10(eta_c)])

    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s")

    return np.array(X_list), np.array(Y_list)


# ── Neural network for histogram -> parameters ─────────────────────────────

class HistogramNet(nn.Module):
    """
    Small MLP: histogram (n_bins) -> (log_bandwidth, log_eta_coupling).
    3-4 hidden layers, 64 units each.
    """
    def __init__(self, n_bins=50, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_bins, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 2))  # 2 outputs
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Training ────────────────────────────────────────────────────────────────

def train_histogram_net(X_train, Y_train, epochs=2000, lr=1e-3, batch_size=16,
                        verbose=True):
    """Train the histogram network."""
    X_t = torch.tensor(X_train, dtype=DTYPE)
    Y_t = torch.tensor(Y_train, dtype=DTYPE)

    n_samples = X_t.shape[0]
    n_bins = X_t.shape[1]

    model = HistogramNet(n_bins=n_bins, hidden_dim=64, n_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    history = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_t[idx]
            Y_batch = Y_t[idx]

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = torch.mean((Y_pred - Y_batch)**2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if verbose and epoch % 200 == 0:
            elapsed = time.time() - t_start
            print(f"    epoch {epoch:5d}/{epochs} | loss {avg_loss:.4e} | {elapsed:.1f}s")

    elapsed = time.time() - t_start
    if verbose:
        print(f"  Training complete in {elapsed:.1f}s")

    return model, history


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_training_loss(history):
    """Plot training loss vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history, 'b-', lw=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('Histogram Network: Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_predictions(model, X_test, Y_test):
    """Plot predicted vs true bandwidth and coupling for test set."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=DTYPE)
    with torch.no_grad():
        Y_pred = model(X_t).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Histogram Network: Test Set Predictions', fontsize=13)

    # Bandwidth
    ax = axes[0]
    true_bw = Y_test[:, 0]
    pred_bw = Y_pred[:, 0]
    ax.plot(true_bw, pred_bw, 'ro', ms=8, alpha=0.7)
    lims = [min(true_bw.min(), pred_bw.min()) - 0.2,
            max(true_bw.max(), pred_bw.max()) + 0.2]
    ax.plot(lims, lims, 'k--', lw=1, label='Perfect prediction')
    ax.set_xlabel('True log10(bandwidth / Hz)')
    ax.set_ylabel('Predicted log10(bandwidth / Hz)')
    ax.set_title('Acceptance Bandwidth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Compute R^2
    ss_res = np.sum((true_bw - pred_bw)**2)
    ss_tot = np.sum((true_bw - true_bw.mean())**2)
    r2_bw = 1 - ss_res / ss_tot
    ax.text(0.05, 0.92, f'R$^2$ = {r2_bw:.3f}', transform=ax.transAxes, fontsize=11)

    # eta_coupling
    ax = axes[1]
    true_eta = Y_test[:, 1]
    pred_eta = Y_pred[:, 1]
    ax.plot(true_eta, pred_eta, 'bo', ms=8, alpha=0.7)
    lims = [min(true_eta.min(), pred_eta.min()) - 0.2,
            max(true_eta.max(), pred_eta.max()) + 0.2]
    ax.plot(lims, lims, 'k--', lw=1, label='Perfect prediction')
    ax.set_xlabel('True log10(eta_coupling)')
    ax.set_ylabel('Predicted log10(eta_coupling)')
    ax.set_title('Coupling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ss_res = np.sum((true_eta - pred_eta)**2)
    ss_tot = np.sum((true_eta - true_eta.mean())**2)
    r2_eta = 1 - ss_res / ss_tot
    ax.text(0.05, 0.92, f'R$^2$ = {r2_eta:.3f}', transform=ax.transAxes, fontsize=11)

    plt.tight_layout()
    return fig, r2_bw, r2_eta


def plot_example_histograms(laser, sld, P_sld):
    """Show eta histograms at different bandwidths with fixed coupling."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('AMZI Splitting Ratio Histograms at Different Bandwidths\n'
                 '(eta_coupling = 0.1, f_rep = 5 GHz)', fontsize=13)

    bandwidths = [100e9, 300e9, 500e9, 1e12, 2e12, 5e12]
    eta_c = 0.1

    for ax, bw in zip(axes.flatten(), bandwidths):
        S_inj = compute_S_inj(laser, sld, P_sld, bw, eta_c)
        counts, bin_edges, eta_raw = generate_eta_histogram(
            laser, S_inj, n_pulses=20000, n_discard=300,
            n_bins=50, seed=77,
        )

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
               alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
        ax.set_xlabel('$\\eta$ = cos$^2$($\\Delta\\phi$/2)')
        ax.set_ylabel('Probability')

        # Format bandwidth label
        if bw >= 1e12:
            bw_str = f'{bw/1e12:.1f} THz'
        else:
            bw_str = f'{bw/1e9:.0f} GHz'
        ax.set_title(f'BW = {bw_str}')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

        # Annotate S_inj and std(eta)
        ax.text(0.95, 0.92, f'S_inj = {S_inj:.1e}\n'
                f'std($\\eta$) = {np.std(eta_raw):.3f}',
                transform=ax.transAxes, fontsize=8,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("  PINN Bandwidth: Learning Acceptance Bandwidth from AMZI Data")
    print("=" * 70)

    # Setup laser and SLD
    laser = DFBLaserParams()
    I_th = laser.threshold_current()
    sld = SLDParams()

    # Compute SLD output power
    sld_result = solve_sld_steady_state(sld, 150e-3)
    P_sld = sld_result['P_out']

    print(f"\n  Laser: 1550 nm DFB, I_th = {I_th*1e3:.1f} mA")
    print(f"  SLD: P_out = {P_sld*1e3:.2f} mW (I_sld = 150 mA)")
    print(f"  Repetition rate: 5 GHz")
    print(f"  Bandwidth range: 100 GHz - 5 THz")
    print(f"  Coupling range: 0.01 - 1.0")

    # ── Warm up numba JIT ───────────────────────────────────────────────
    print("\n  Warming up numba JIT compiler...")
    t0 = time.time()
    _ = simulate_pulses(
        100, 10, 100, 1e-12,
        0.9 * I_th, 5.0 * I_th, 30e-12, 7e-12,
        laser.V, laser.Gamma, laser.v_g, laser.a, laser.N_tr, laser.epsilon,
        laser.A, laser.B, laser.C, laser.tau_p, laser.beta_sp, laser.alpha_H, q,
        0.0, 0)
    print(f"  JIT compiled in {time.time()-t0:.1f}s")

    # ── Step 1: Generate example histograms (figure 3) ──────────────────
    print("\n" + "-" * 70)
    print("  Generating example histograms...")
    print("-" * 70)
    fig3 = plot_example_histograms(laser, sld, P_sld)
    fig3.savefig(os.path.join(OUT_DIR, 'example_histograms.png'), dpi=150,
                 bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: images/pinn_bandwidth/example_histograms.png")

    # ── Step 2: Generate training data ──────────────────────────────────
    print("\n" + "-" * 70)
    print("  Generating training data (grid of bandwidth x coupling)...")
    print("-" * 70)
    t_gen_start = time.time()

    X_train, Y_train, bw_range, eta_range = generate_training_data(
        laser, sld, P_sld,
        n_bw=10, n_eta=8,         # 80 combinations
        n_pulses_train=15000,     # coarser simulation for speed
        n_bins=50,
    )
    t_gen = time.time() - t_gen_start
    print(f"  Training data: {X_train.shape[0]} histograms generated in {t_gen:.1f}s")

    # ── Step 3: Generate test data ──────────────────────────────────────
    print("\n  Generating test data (held-out random combinations)...")
    X_test, Y_test = generate_test_data(
        laser, sld, P_sld,
        n_test=15,
        n_pulses_test=30000,      # more pulses for cleaner test histograms
        n_bins=50,
    )
    print(f"  Test data: {X_test.shape[0]} histograms")

    # ── Step 4: Train neural network ────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Training histogram network...")
    print("-" * 70)

    model, history = train_histogram_net(
        X_train, Y_train,
        epochs=2000, lr=1e-3, batch_size=16,
    )

    # Plot 1: Training loss
    fig1 = plot_training_loss(history)
    fig1.savefig(os.path.join(OUT_DIR, 'training_loss.png'), dpi=150,
                 bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved: images/pinn_bandwidth/training_loss.png")

    # ── Step 5: Evaluate on test set ────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Evaluating on test set...")
    print("-" * 70)

    fig2, r2_bw, r2_eta = plot_predictions(model, X_test, Y_test)
    fig2.savefig(os.path.join(OUT_DIR, 'predictions.png'), dpi=150,
                 bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: images/pinn_bandwidth/predictions.png")

    # ── Summary ─────────────────────────────────────────────────────────
    model.eval()
    X_t = torch.tensor(X_test, dtype=DTYPE)
    with torch.no_grad():
        Y_pred = model(X_t).numpy()

    print("\n" + "=" * 70)
    print("  SUMMARY: Test Set Results")
    print("=" * 70)
    print(f"\n  R^2 (bandwidth): {r2_bw:.4f}")
    print(f"  R^2 (coupling):  {r2_eta:.4f}")

    # Per-sample errors
    bw_err = np.abs(10**Y_pred[:, 0] - 10**Y_test[:, 0]) / 10**Y_test[:, 0] * 100
    eta_err = np.abs(10**Y_pred[:, 1] - 10**Y_test[:, 1]) / 10**Y_test[:, 1] * 100

    print(f"\n  Bandwidth error: median {np.median(bw_err):.1f}%, "
          f"mean {np.mean(bw_err):.1f}%, max {np.max(bw_err):.1f}%")
    print(f"  Coupling error:  median {np.median(eta_err):.1f}%, "
          f"mean {np.mean(eta_err):.1f}%, max {np.max(eta_err):.1f}%")

    print(f"\n  {'#':<4} {'True BW':>12} {'Pred BW':>12} {'Err%':>6} "
          f"{'True eta':>10} {'Pred eta':>10} {'Err%':>6}")
    print(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*6}")
    for i in range(len(Y_test)):
        true_bw = 10**Y_test[i, 0]
        pred_bw = 10**Y_pred[i, 0]
        true_eta = 10**Y_test[i, 1]
        pred_eta = 10**Y_pred[i, 1]
        e_bw = abs(pred_bw - true_bw) / true_bw * 100
        e_eta = abs(pred_eta - true_eta) / true_eta * 100
        print(f"  {i+1:<4} {true_bw:>12.2e} {pred_bw:>12.2e} {e_bw:>5.1f}% "
              f"{true_eta:>10.4f} {pred_eta:>10.4f} {e_eta:>5.1f}%")

    print(f"\n  All figures saved to: images/pinn_bandwidth/")
    print("=" * 70)
