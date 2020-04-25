"""Logistic regression example
"""
import os
import re
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
    experiments_cfg = file_io.load_yaml(experiments_path)

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

        rendered_config = model_utils.render_template(
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

@click.command()
@click.option('--platform', '-p', default='slurm', type=str, help='Platform running experiments')
def run(platform):
    """Controller function which creates the server and starts off the training

    Args:
        n_workers (int): No. of clients to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    # Load in config to setup model
    folder_name = _create_experiments()
    folder_name = 'config'/folder_name
    # n_workers = [8, 32, 64, 128]
    # n_workers = [8, 16, 32, 64]
    # n_workers = [64]
    project_dir = Path(__file__).resolve().parents[1]
    launch_script_path = str(project_dir/'scripts/slurm_launch.sh')

    if platform == 'slurm':
        for fname in folder_name.iterdir():
            # print(fname)
            e = file_io.load_yaml(fname)
            model_version = e["model"]["version"]
            n_worker = e["distribute"]["n_workers"]
            # print(model_version)
            # print(n_workers)
            # for n_worker in n_workers:
            result = subprocess.run([
                'sbatch',
                '-n',
                str(n_worker),
                launch_script_path,
                '-v',
                'DEBUG',
                '-m',
                model_version,
                '-c',
                str(fname)
            ])

            slurm_jobid_match = re.search('(?<=batch job ).+', result.stdout)
            if slurm_jobid_match:
                e['slurm_jobid'] = slurm_jobid_match.group()
                file_io.export_yaml(e, fname)

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
