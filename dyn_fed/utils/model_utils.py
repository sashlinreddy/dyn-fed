"""All model utility functions
"""
import os
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from dyn_fed.utils import string_utils

logger = logging.getLogger("dfl.utils.model_utils")

def encode_run_name(n_workers, config):
    """Encode model run name given it's config. Useful for tracking experiments

    Args:
        n_workers (int): No. of clients in experiment
        config (dict): Configuration for model

    Returns:
        encode_name (str): Encoded experiment name
    """
    encode_vars = [
        "n_workers", "scenario", "quantize",
        "aggregate_mode", "interval", "mode", "noniid",
        "unbalanced", "learning_rate"
    ]

    global_cfg = {"n_workers": n_workers}
    global_cfg.update(config["data"])
    global_cfg.update(config["executor"])
    global_cfg.update(config["optimizer"])
    global_cfg.update(config["comms"])
    global_cfg.update(config["distribute"])
    encode_name = string_utils.dict_to_str(global_cfg, encode_vars)

    if "LOGDIR" in os.environ:
        data_name = Path(config.data.name)
        model_type = Path(config.model.type)
        opt_name = Path(config.optimizer.name)
        logdir = os.environ["LOGDIR"]/data_name/model_type/opt_name/encode_name
        logdir.mkdir(parents=True, exist_ok=True)
        os.environ["LOGDIR"] = str(logdir)

    return encode_name

def render_template(template_dir, template_name, **kwargs):
    """Render jinja template
    """
    if not isinstance(template_dir, str):
        template_dir = str(template_dir)
    file_loader = FileSystemLoader(template_dir)
    logger.debug(f"Template dir={template_dir}")
    env = Environment(loader=file_loader)
    template = env.get_template(template_name)

    rendered = template.render(
        **kwargs
    )

    return rendered
