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

@click.command()
@click.option('--platform', '-p', default='slurm', type=str, help='Platform running experiments')
@click.option('--dirname', '-d', default=None, type=str, help='Config directory')
def run(platform, dirname):
    """Controller function which creates the server and starts off the training

    Args:
        n_workers (int): No. of clients to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    # Load in config to setup model
    project_dir = Path(__file__).resolve().parents[1]
    if dirname is None:
        folder_name = model_utils.create_experiments()
        folder_name = project_dir/'config'/folder_name
    else:
        folder_name = project_dir/'config'/dirname

    # n_workers = [8, 32, 64, 128]
    # n_workers = [8, 16, 32, 64]
    # n_workers = [64]
    launch_script_path = str(project_dir/'scripts/slurm_launch.sh')

    if platform == 'slurm':
        for fname in folder_name.iterdir():
            # print(fname)
            e = file_io.load_yaml(fname, to_obj=False)
            model_version = e["model"]["version"]
            n_workers = e["distribute"]["n_workers"]
            # print(model_version)
            # print(n_workers)
            # for n_worker in n_workers:
            result = subprocess.run([
                'sbatch',
                '-n',
                str(n_workers),
                launch_script_path,
                '-v',
                'DEBUG',
                '-m',
                model_version,
                '-c',
                str(fname)
                ],
                capture_output=True
            )

            slurm_jobid_match = re.search('(?<=batch job ).+', result.stdout.decode())
            if slurm_jobid_match:
                e['slurm'] = {"jobid": int(slurm_jobid_match.group())}
                file_io.export_yaml(e, fname)

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
