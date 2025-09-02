from pathlib import Path
import pickle
import numpy as np
from .myfun import GP_training

def train_or_load_gpr(data_dir: Path, ckpt_dir: Path):
    """Load pre-trained GP from checkpoints, or train if you expose data + labels."""
    gp_path = ckpt_dir / "GPexperiment.dump"
    if gp_path.exists():
        return pickle.load(open(gp_path, "rb"))

    # If you decide to expose training, implement:
    # X_train, Y_train = ...
    # models = []
    # for i in range(Y_train.shape[1]):
    #     models.append(GP_training(X_train, Y_train[:, i], Option=1))
    # pickle.dump(models, open(gp_path, "wb"))
    # return models
    raise FileNotFoundError("Missing GP checkpoint. Provide checkpoints/GPexperiment.dump")
