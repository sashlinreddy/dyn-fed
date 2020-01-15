"""Logistic regression example
"""
import os
from datetime import datetime
from pathlib import Path

import click
import numpy as np
# For reproducibility
# pylint: disable=wrong-import-position
np.random.seed(42)

from fault_tolerant_ml.lib.io import file_io

from fault_tolerant_ml.utils import model_utils

@click.command()
def run():
    """Controller function which creates the master and starts off the training

    Args:
        n_workers (int): No. of workers to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    # Load in config to setup model
    config_dir = 'config/'
    config_path = 'config/template.yml'
    experiments_path = 'config/experiments.yml'
    if 'PROJECT_DIR' in os.environ:
        config_dir = Path(os.environ["PROJECT_DIR"])/'config'
        config_path = Path(os.environ['PROJECT_DIR'])/config_path
        experiments_path = Path(os.environ['PROJECT_DIR'])/'config/experiments.yml'

    experiments_cfg = file_io.load_model_config(experiments_path)

    # data_folders = experiments_cfg["executor"]["shared_folder"]
    # comm_modes = experiments_cfg["comms"]["mode"]
    # agg_modes = experiments_cfg["distribute"]["aggregate_mode"]
    # noniids = experiments_cfg["distribute"]["aggregate_mode"]
    # agg_modes = experiments_cfg["distribute"]["aggregate_mode"]

    import itertools
    keys, values = zip(*experiments_cfg.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    folder_name = config_dir/Path(datetime.now().strftime("%Y%m%d-%H%M"))
    folder_name.mkdir(exist_ok=True)

    counter = 0
    # Write all config to folder
    for i, experiment in enumerate(experiments):
        rendered_config = model_utils.render_template(
            config_dir,
            'template.yml',
            n_iterations=100,
            comm_mode=experiment.get('mode', 0),
            interval=1,
            agg_mode=experiment.get('aggregate_mode', 0),
            data_dir=f"\"{experiment.get('data_dir'), 'data/mnist/'}\"",
            noniid=experiment.get('noniid', 0),
            unbalanced=experiment.get('unbalanced', 0)
        )
        counter = i + 1
        with open(f'{folder_name}/config-{i+1}.yml', 'w') as f:
            f.write(rendered_config)

    print(f"Written {counter} experiments to config/{folder_name} folder")

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
