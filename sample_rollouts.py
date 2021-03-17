"""This script shows how to sample new rollouts from a variational-trained HGN.
"""
from hamiltonian_generative_network import HGN
from utilities.integrator import Integrator
from utilities.loader import load_hgn
from train import _read_config, _merge_configs
import torch

if __name__=="__main__":
    model_to_load = "saved_models/default_offline"

    _train_config = _read_config("experiment_params/train_config_default.yaml")
    if _args.dataset_path is None:  # Will use the dataset config file (or default if not given)
        _dataset_config_file = DEFAULT_DATASET_CONFIG_FILE if _args.dataset_config is None else \
            _args.dataset_config[0]
        _dataset_config = _read_config("datasets/ThreeSpring/parameters.yaml")
        _config = _merge_configs(_train_config, _dataset_config)

    dtype = torch.__getattribute__(_config["networks"]["dtype"])

    integrator = Integrator(delta_t=0.125, method="Leapfrog")
    hgn = load_hgn(params=_config, device="cpu", dtype=dtype)  # If going to load, no need to specify networks
    hgn.load(model_to_load)

    # Sample a rollout of n_steps
    prediction = hgn.get_random_sample(n_steps=50, img_shape=(32, 32))
    prediction.visualize()

