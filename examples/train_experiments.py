"""Logistic regression example
"""
import os
from datetime import datetime
from pathlib import Path
import itertools
import subprocess

import click

from dyn_fed.lib.io import file_io

from dyn_fed.utils import model_utils

def _create_experiments():
    """Create experiments for dfl
    """
    config_dir = 'config/'
    config_path = 'config/template.yml'
    experiments_path = 'config/experiments.yml'
    if 'PROJECT_DIR' in os.environ:
        config_dir = Path(os.environ["PROJECT_DIR"])/'config'
        config_path = Path(os.environ['PROJECT_DIR'])/config_path
        experiments_path = Path(os.environ['PROJECT_DIR'])/'config/experiments.yml'

    # Load all hyperparameters
    experiments_cfg = file_io.load_model_config(experiments_path)

    # Generate permutations
    keys, values = zip(*experiments_cfg.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    folder_name = Path(datetime.now().strftime("%Y%m%d-%H%M"))
    folder_path = config_dir/folder_name
    folder_path.mkdir(exist_ok=True)

    counter = 0
    # Write all config to folder
    for i, experiment in enumerate(experiments):
        rendered_config = model_utils.render_template(
            config_dir,
            'template.yml',
            model_type=experiment.get('model_type', 'logistic'),
            n_iterations=experiment.get('n_iterations', 100),
            check_overfitting=experiment.get('check_overfitting', False),
            data_name=experiment.get('data_name', 'mnist'),
            noniid=experiment.get('noniid', 0),
            unbalanced=experiment.get('unbalanced', 0),
            optimizer=experiment.get('optimizer', 'sgd'),
            comm_mode=experiment.get('mode', 0),
            interval=experiment.get('interval', 1),
            agg_mode=experiment.get('aggregate_mode', 0),
            delta_threshold=experiment.get('delta_threshold', 0.0),
            data_dir=experiment.get('shared_folder', 'data/mnist/')
        )
        counter = i + 1
        with open(f'{folder_path}/config-{i+1}.yml', 'w') as f:
            f.write(rendered_config)

    print(f"Written {counter} experiments to config/{folder_name} folder")

    return folder_name

@click.command()
@click.option('--platform', '-p', default='slurm', type=str, help='Platform running experiments')
def run(platform):
    """Controller function which creates the master and starts off the training

    Args:
        n_workers (int): No. of workers to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    # Load in config to setup model
    folder_name = _create_experiments()
    folder_name = 'config'/folder_name
    # n_workers = [8, 32, 64, 128]
    # n_workers = [8, 16, 32]
    n_workers = [64]

    if platform == 'slurm':
        for fname in folder_name.iterdir():
            print(fname)
            for n_worker in n_workers:
                subprocess.run([
                    'sbatch',
                    '-n',
                    str(n_worker),
                    '/home-mscluster/sreddy/fault-tolerant-ml/scripts/slurm_launch.sh',
                    '-v',
                    'DEBUG',
                    '-m',
                    'LOG2',
                    '-c',
                    str(fname)
                ])

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
