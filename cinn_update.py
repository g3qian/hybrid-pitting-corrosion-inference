from pathlib import Path
import pickle
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer

def _build_summary_net():
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(8, kernel_size=2, strides=2, activation='relu'),
        tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='elu'),
    ])

def build_cinn(n_params: int):
    return AmortizedPosterior(
        InvertibleNetwork(num_params=n_params, num_coupling_layers=10),
        _build_summary_net(),
        name="HybridPitting"
    )

def train_cinn(data_dir: Path, ckpt_dir: Path, weights_path: Path):
    with open(data_dir / "input_beta.pkl", "rb") as f:
        para_all = pickle.load(f)
    with open(data_dir / "output_beta.pkl", "rb") as f:
        pit_depth_all = pickle.load(f)

    pit_depth_all = np.transpose(pit_depth_all, (0, 2, 1))
    mask = ~np.all(pit_depth_all == 0, axis=(1,2))
    pit_depth = pit_depth_all[mask]
    para = para_all[mask]

    scaler = preprocessing.StandardScaler().fit(para)
    para_scaled = scaler.transform(para)
    train_disp = tf.convert_to_tensor(pit_depth, dtype=tf.float32)
    train_para = tf.convert_to_tensor(para_scaled, dtype=tf.float32)

    class GenModel:
        def __call__(self, batch_size): return train_disp, train_para
    def configurator(batch_size): return {"summary_conditions": train_disp, "parameters": train_para}

    amortizer = build_cinn(n_params=para.shape[1])
    trainer = Trainer(amortizer=amortizer, generative_model=GenModel(), configurator=configurator)
    trainer.train_online(epochs=1500, iterations_per_epoch=10, batch_size=64)

    amortizer.save_weights(str(weights_path))
    return amortizer, scaler

def load_cinn(n_params: int, weights_path: Path):
    amortizer = build_cinn(n_params)
    amortizer.load_weights(str(weights_path))
    # You’ll need to also load the fitted scaler when you distribute it,
    # or rebuild the same scaler if you expose the training data.
    # For public release, consider shipping a pre-fitted scaler.pkl.
    # Here we assume a scaler.pkl exists:
    scaler_pkl = weights_path.parent / "scaler.pkl"
    if scaler_pkl.exists():
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
    else:
        raise FileNotFoundError("Missing checkpoints/scaler.pkl")
    return amortizer, scaler
