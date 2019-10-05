"""All model utility functions
"""
import os

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
        "aggregate_mode", "interval", "mode"
    ]

    global_cfg = {"n_workers": n_workers}
    global_cfg.update(config["executor"])
    global_cfg.update(config["optimizer"])
    global_cfg.update(config["comms"])
    global_cfg.update(config["distribute"])
    encode_name = string_utils.dict_to_str(global_cfg, encode_vars)

    if "LOGDIR" in os.environ:
        logdir = os.path.join(os.environ["LOGDIR"], encode_name)
        if not os.path.exists(logdir):
            try:
                os.mkdir(logdir)
            except FileExistsError:
                pass
        os.environ["LOGDIR"] = logdir

    return encode_name
