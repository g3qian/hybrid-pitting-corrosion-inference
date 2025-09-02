from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def _remove_nans_from_rows(df):
    return [row[~np.isnan(row)] for row in df.to_numpy()]

def _area_to_radius(a):  # a in mm^2 -> r in mm
    return np.sqrt(a / np.pi)

def _concat_selected_slices(radius_dict, selection):
    # selection like {'025':[2], '050':[2], '100':[0], '200':[2], '300':[2]}
    out = {}
    for k, idxs in selection.items():
        pool = []
        for i in idxs:
            if i < len(radius_dict[k]):
                pool.append(radius_dict[k][i])
        out[k] = np.concatenate(pool) if pool else np.array([])
    return out

def load_measurements_kde(data_dir: Path):
    """Return KDEs across time slices on a common x-grid (0..1.3, 1000 pts)."""
    df_area = pd.read_excel(data_dir / "si_area_values.xlsx", header=None)
    # Group by time keys and convert area->radius
    radius = {
        '025': [_area_to_radius(r) for r in _remove_nans_from_rows(df_area.iloc[:11])],
        '050': [_area_to_radius(r) for r in _remove_nans_from_rows(df_area.iloc[12:32])],
        '100': [_area_to_radius(r) for r in _remove_nans_from_rows(df_area.iloc[33:50])],
        '200': [_area_to_radius(r) for r in _remove_nans_from_rows(df_area.iloc[51:65])],
        '300': [_area_to_radius(r) for r in _remove_nans_from_rows(df_area.iloc[66:])],
    }
    selection = {'025':[2], '050':[2], '100':[0], '200':[2], '300':[2]}
    chosen = _concat_selected_slices(radius, selection)

    x = np.linspace(0.0, 1.3, 1000)
    kdes = []
    for key in ['025','050','100','200','300']:
        vals = chosen[key]
        kde = gaussian_kde(vals)
        kdes.append(kde(x))
    return np.stack(kdes, axis=0), x
