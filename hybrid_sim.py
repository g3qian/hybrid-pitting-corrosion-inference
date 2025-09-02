import numpy as np
from scipy.stats import beta as beta_dist
from .myfun import GP_predict

def number_of_pits(alpha, b, total_steps=201, horizon=250, n_pits=1000):
    """Cumulative initiations over time based on Beta(α,b) scaled to horizon; discard > total_steps."""
    init_times = beta_dist.rvs(alpha, b, size=n_pits)
    init_steps = np.round(init_times * (horizon - 1)).astype(int)
    init_steps = init_steps[init_steps < total_steps]
    return np.array([np.sum(init_steps <= t) for t in range(total_steps)])

def run_hybrid_simulation(alpha, b, theta1, theta2, gp_models, total_steps=201, n_pits=500):
    """Simulate multi-pit depths over time with GP surrogate for single-pit growth."""
    init_times = beta_dist.rvs(alpha, b, size=n_pits)
    init_steps = np.clip(np.round(init_times * (total_steps - 1)).astype(int), 0, total_steps - 1)

    tvec = np.arange(total_steps) * 0.005
    depth = np.ones((total_steps, n_pits)) * 0.037
    Xbuf = np.empty((0,3))  # on-demand batching if you later vectorize

    for n in range(n_pits):
        L = total_steps - init_steps[n]
        X = np.column_stack([
            np.full(L, theta1),
            np.full(L, theta2),
            tvec[:L]
        ])
        y_mean, _ = GP_predict(n_important=2, GP_SVD=gp_models, Xtrain=X, Option=1)
        y = y_mean.T[:,0]  # first output channel is depth
        depth[init_steps[n]:, n] = y

    return depth - 0.037  # net increase
