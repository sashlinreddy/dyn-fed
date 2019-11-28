"""All model utility functions
"""
import os
from pathlib import Path

from fault_tolerant_ml.utils import string_utils

def encode_run_name(n_workers, config):
    """Encode model run name given it's config. Useful for tracking experiments

    Args:
        n_workers (int): No. of workers in experiment
        config (dict): Configuration for model

    Returns:
        encode_name (str): Encoded experiment name
    """
    encode_vars = [
        "n_workers", "scenario", "remap", "quantize",
        "send_gradients", "mu_g", "n_most_rep", "overlap",
        "aggregate_mode", "interval", "mode", "noniid",
        "unbalanced"
    ]

    global_cfg = {"n_workers": n_workers}
    global_cfg.update(config["executor"])
    global_cfg.update(config["optimizer"])
    global_cfg.update(config["comms"])
    global_cfg.update(config["distribute"])
    encode_name = string_utils.dict_to_str(global_cfg, encode_vars)

    if "LOGDIR" in os.environ:
        data_name = Path(Path(config["executor"]["shared_folder"]).stem)
        logdir = os.environ["LOGDIR"]/data_name/encode_name
        logdir.mkdir(parents=True, exist_ok=True)
        os.environ["LOGDIR"] = str(logdir)

    return encode_name
