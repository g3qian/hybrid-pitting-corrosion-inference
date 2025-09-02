#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid multiple-pit inference pipeline
Steps:
1) Extract & analyze measurement data
2) GPR surrogate modeling for single-pit growth
3) Hybrid simulation for multiple pits
4) cINN model updating with measurement data
5) Predict with the updated model

Default: loads pre-trained cINN weights from checkpoints/.
Use --train to fit cINN (requires private data PKLs).
"""

from pathlib import Path
import argparse
import numpy as np

from .data_io import load_measurements_kde                       # (1)
from .gpr_surrogate import train_or_load_gpr                     # (2)
from .hybrid_sim import run_hybrid_simulation, number_of_pits    # (3)
from .cinn_update import build_cinn, load_cinn, train_cinn       # (4)
from .predict import posterior_predict_pdf, summarize_percentile # (5)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
WEIGHTS = CKPT / "weights_beta"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Retrain cINN from scratch (needs private PKLs).")
    p.add_argument("--samples", type=int, default=10000, help="Posterior samples for prediction.")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs"), help="Output folder for figures/tables.")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # (1) Measurement KDEs (radius PDFs over a fixed grid), and any summary stats you need
    kde_exp, x_grid = load_measurements_kde(DATA)
    # kde_exp shape expected: (5, 1000) for 0.25h, 0.5h, 1h, 2h, 3h

    # (2) Train or load GPR surrogate for single-pit growth (uses your myfun.py interface)
    gp = train_or_load_gpr(DATA, CKPT)

    # (3) Hybrid multi-pit simulation utilities available if needed for diagnostics
    # Example (disable by default):
    # sim_depth = run_hybrid_simulation(alpha=4.5, b=2.2, theta_1=0.008, theta_2=0.05, gp=gp)

    # (4) cINN updating: train (private) or load (public)
    if args.train:
        amortizer, scaler = train_cinn(DATA, CKPT, weights_path=WEIGHTS)
    else:
        amortizer, scaler = load_cinn(n_params=4, weights_path=WEIGHTS)

    # Prepare observation to condition on: use first 4 time-points of measurement KDEs
    obs = kde_exp[:4, :]             # (4, 1000)
    obs = obs.reshape(1, 1000, 4)    # (batch=1, len=1000, channels=4)

    # (5) Posterior sampling and prediction at 3h (last time point)
    post_samples = amortizer.sample({"summary_conditions": obs}, n_samples=args.samples).numpy()
    post_samples = scaler.inverse_transform(post_samples)  # columns: [lambda, beta, theta1, theta2]

    # Predict PDFs at 3h for each posterior draw via the hybrid simulator (vectorized batching inside)
    pred_pdfs = posterior_predict_pdf(post_samples, gp, x_grid=x_grid)  # shape: (N, 1000)

    # Compare with measurement at 3h (kde_exp[-1])
    p90_pred_mean, p90_pred_ci = summarize_percentile(pred_pdfs, x_grid, q=0.90)
    p90_meas = summarize_percentile(kde_exp[-1][None, :], x_grid, q=0.90)  # same util works for 1-row

    np.save(outdir / "post_samples.npy", post_samples)
    np.save(outdir / "pred_pdfs.npy", pred_pdfs)

    print(f"[ok] Finished. Outputs in: {outdir}")
    print(f"Predicted 90th percentile at 3h (mean, 95% CI): {p90_pred_mean:.4f}, {p90_pred_ci}")
    print(f"Measured 90th percentile at 3h: {p90_meas[0]:.4f}")

if __name__ == "__main__":
    main()
