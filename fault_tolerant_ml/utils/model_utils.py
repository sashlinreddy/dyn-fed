import os

from fault_tolerant_ml.utils import string_utils

def encode_run_name(n_workers, config):
    encode_vars = [
        "n_workers", "scenario", "remap", "quantize",
        "comm_period", "send_gradients", "mu_g", "n_most_rep"
    ]

    global_cfg = {"n_workers": n_workers}
    global_cfg.update(config["executor"])
    global_cfg.update(config["optimizer"])
    encode_name = string_utils.dict_to_str(global_cfg, choose=encode_vars)

    if "LOGDIR" in os.environ:
        logdir = os.path.join(os.environ["LOGDIR"], encode_name)
        if not os.path.exists(logdir):
            try:
                os.mkdir(logdir)
            except FileExistsError:
                pass
        os.environ["LOGDIR"] = logdir

    return encode_name
