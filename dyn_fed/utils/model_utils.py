"""All model utility functions
"""
import os
import logging
from pathlib import Path
from datetime import datetime
import itertools

from jinja2 import Environment, FileSystemLoader

from dyn_fed.utils import string_utils
from dyn_fed.lib.io import file_io

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
        "unbalanced", "learning_rate", "n_iterations"
    ]

    global_cfg = {"n_workers": n_workers}
    global_cfg.update(config["model"])
    global_cfg.update(config["data"])
    global_cfg.update(config["executor"])
    global_cfg.update(config["optimizer"])
    global_cfg.update(config["comms"])
    global_cfg.update(config["distribute"])
    encode_name = string_utils.dict_to_str(global_cfg, encode_vars)

    if config.comms.mode == 3:
        encode_name += "-" + str(config.distribute.delta_threshold)

    if "LOGDIR" in os.environ:
        data_name = Path(config.data.name)
        model_type = Path(config.model.type)
        opt_name = Path(config.optimizer.name)
        date = Path(datetime.now().strftime("%Y%m%d"))
        logdir = os.environ["LOGDIR"]/date/data_name/model_type/opt_name/encode_name
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

def gen_exp_perms(experiments_cfg):
    """Generate experiment permutations
    """
    # Generate permutations
    keys, values = zip(*experiments_cfg.items())
    # experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    experiments = []
    for v in itertools.product(*values):
        d = dict(zip(keys, v))
        # Ignoring invalid experiments
        # Mode > 0 doesn't require periodic interval
        # NNs run for 50 iterations
        if (d.get("mode") > 0 and d.get("interval") > 1) or \
        (d.get("model_type").find("nn1") >= 0 and d.get("interval") in [20, 100]):
            continue
        # Exclude threshold > 0 for mode != 3
        # For now adam optimizer delta threshold no work
        elif (d.get("mode") != 3 and d.get("delta_threshold") > 0.0):
            continue
        elif ((d.get('mode') == 3) and (d.get('optimizer') == 'adam') \
            and (d.get("delta_threshold") < 2.0)):
            continue
        else:
            experiments.append(d)

    return experiments

def create_experiments(project_dir=None):
    """Create experiments for dfl
    """
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[2]
    config_dir = project_dir/'config/'
    config_path = project_dir/'config/template.yml'
    experiments_path = project_dir/'config/experiments.yml'

    # Load all hyperparameters
    experiments_cfg = file_io.load_yaml(experiments_path)

    experiments = gen_exp_perms(experiments_cfg)

    folder_name = Path(datetime.now().strftime("%Y%m%d-%H%M"))
    folder_path = config_dir/folder_name
    folder_path.mkdir(exist_ok=True)

    counter = 0
    # Write all config to folder
    for i, experiment in enumerate(experiments):
        # Validate comm interval
        n_iterations = experiment.get('n_iterations', 100) if \
                experiment.get('model_type') == "logistic" else 50
        interval = experiment.get('interval', 1) if \
                experiment.get('interval', 1) <= n_iterations else n_iterations
        learning_rate = 0.001 if \
                experiment.get('optimizer', 'sgd') == 'adam' else 0.01

        rendered_config = render_template(
            config_dir,
            'template.yml',
            model_version=experiment.get('model_version', 'TF'),
            model_type=experiment.get('model_type', 'logistic'),
            n_iterations=n_iterations,
            check_overfitting=experiment.get('check_overfitting', False),
            data_name=experiment.get('data_name', 'mnist'),
            noniid=experiment.get('noniid', 0),
            unbalanced=experiment.get('unbalanced', 0),
            learning_rate=learning_rate,
            optimizer=experiment.get('optimizer', 'sgd'),
            comm_mode=experiment.get('mode', 0),
            interval=interval,
            n_workers=experiment.get('n_workers', 8),
            agg_mode=experiment.get('aggregate_mode', 0),
            delta_threshold=experiment.get('delta_threshold', 0.0),
            data_dir=experiment.get('shared_folder', 'data/mnist/')
        )
        counter = i + 1
        with open(f'{folder_path}/config-{i+1}.yml', 'w') as f:
            f.write(rendered_config)

    print(f"Written {counter} experiments to config/{folder_name} folder")
    return folder_name
