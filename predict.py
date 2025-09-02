import numpy as np
from scipy.interpolate import interp1d
from .hybrid_sim import run_hybrid_simulation
from .myfun import GP_predict

def _pdf_at_3h(alpha, b, theta1, theta2, gp, x_grid):
    depth = run_hybrid_simulation(alpha, b, theta1, theta2, gp_models=gp)
    last = depth[200, :]             # values at 3h
    pos = last[last > 0]
    if len(pos) < 5:  # too few
        return np.zeros_like(x_grid)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(pos)
    return kde(x_grid)

def posterior_predict_pdf(post_samples, gp, x_grid):
    """Predict 3h PDFs for each posterior draw."""
    out = np.zeros((post_samples.shape[0], x_grid.size))
    for i, p in enumerate(post_samples):
        a, b, t1, t2 = p
        out[i, :] = _pdf_at_3h(a, b, t1, t2, gp, x_grid)
    return out

def summarize_percentile(pdfs, x_grid, q=0.90):
    """Return (mean, 95% CI) of the q-quantile across a set of PDFs."""
    qs = []
    dx = x_grid[1] - x_grid[0]
    for pdf in pdfs:
        area = np.trapz(pdf, x_grid)
        if area <= 0: continue
        cdf = np.cumsum(pdf / area) * dx
        inv = interp1d(cdf, x_grid, bounds_error=False, fill_value=(x_grid[0], x_grid[-1]))
        qs.append(float(inv(q)))
    qs = np.array(qs)
    if qs.size == 1:
        return qs[0], (qs[0], qs[0])
    return float(np.mean(qs)), (float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5)))
