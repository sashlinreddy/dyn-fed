"""Logistic regression example
"""
import logging
import os
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
# For reproducibility
# pylint: disable=wrong-import-position
np.random.seed(42)

from dyn_fed.distribute.strategy import MasterWorkerStrategyV2
from dyn_fed.data import mnist, fashion_mnist

from dyn_fed.lib.io import file_io
from dyn_fed.utils import model_utils, setup_logger

from ft_models import LogisticRegressionV2

def train(train_dataset, test_dataset, strategy, config):
    """Perform training session
    """
    logger = logging.getLogger('dfl.train')
    opt_cfg = config.get('optimizer')

    if config.model.type == "logistic":
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(10, activation="sigmoid")
        ])
    elif config.model.type == "nn1":
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation="sigmoid")
        ])

    if opt_cfg.get('name') == 'sgd':
        # Define optimizer
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=opt_cfg.get('learning_rate')
        )
    elif opt_cfg.get("name") == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=opt_cfg.get('learning_rate')
        )

    logger.debug("Running strategy")

    strategy.run(
        model,
        optimizer,
        train_dataset,
        test_dataset=test_dataset
    )

@click.command()
@click.argument('n_workers', type=int)
@click.option('--role', '-r', default="client", type=str)
@click.option('--verbose', '-v', default="INFO", type=str)
@click.option('--identity', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
@click.option('--add', '-a', default=0, type=int)
@click.option('--config', '-c', default='config/config.yml', type=str)
def run(n_workers, role, verbose, identity, tmux, add, config):
    """Controller function which creates the server and starts off the training

    Args:
        n_workers (int): No. of clients to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    if not "SLURM_JOBID" in os.environ:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        pass # Not doing anything for now in terms of flushing logs
        # from dyn_fed.lib.io.file_io import flush_dir
        # _ = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        # flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    # Load in config to setup model
    config_path = config
    if 'PROJECT_DIR' in os.environ:
        config_path = Path(os.environ['PROJECT_DIR'])/config_path
        
    cfg = file_io.load_model_config(config_path)

    data_cfg = cfg["data"]
    executor_cfg = cfg['executor']
    executor_cfg.update(cfg['distribute'])
    executor_cfg.update(cfg['comms'])

    # Create identity
    d_identity: int = 0

    if tmux:
        d_identity = identity = int(identity[1:]) if identity != "" else None
    else:
        d_identity = int(identity) if identity != "" else None
        if add:
            d_identity += 1000

    executor_cfg['identity'] = d_identity

    if 'PROJECT_DIR' in os.environ:
        executor_cfg['shared_folder'] = Path(os.environ['PROJECT_DIR'])/executor_cfg['shared_folder']
        executor_cfg['config_folder'] = Path(os.environ['PROJECT_DIR'])/executor_cfg['config_folder']

    # Encode run name for logs
    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)

    train_dataset = None
    test_dataset = None

    data_name = Path(data_cfg["name"])

    if role == "server":
        # Setup logger
        logger = setup_logger(level=verbose)

        slurm_jobid = os.environ.get("SLURM_JOBID", None)

        if slurm_jobid is not None:
            logger.info(f"SLURM_JOBID={slurm_jobid}")

        # Master reads in data
        if str(data_name) == "fashion-mnist":
            logger.info("Dataset: Fashion-MNist")
            X_train, y_train, X_test, y_test = fashion_mnist.load_data(
                noniid=cfg.data.noniid
            )
            logger.info(f"Dataset={data_cfg.name}")

        elif str(data_name) == 'mnist':
            # Get data
            X_train, y_train, X_test, y_test = mnist.load_data(
                noniid=data_cfg.noniid
            )

            logger.info(f"Dataset={data_cfg.name}")
            
        else:
            raise Exception("Please enter valid dataset")

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = (
                Path(executor_cfg["tf_dir"])/data_name/f"{encoded_run_name}/server"
            )
        train_dataset = (X_train, y_train)
        test_dataset = (X_test, y_test)

    else:
        logger = setup_logger(filename=f'log-client-{d_identity}.log', level=verbose)

        if cfg.model.check_overfitting:
            if cfg.data.name == "mnist":
                logger.info("Dataset: MNist, test set only")
                test_dataset = mnist.load_data(
                    noniid=cfg.data.noniid,
                    test_only=True
                )
            elif cfg.data.name == "fashion-mnist":
                logger.info("Dataset: Fashion-MNist, test set only")
                test_dataset = fashion_mnist.load_data(
                    noniid=cfg.data.noniid,
                    test_only=True
                )

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = Path(executor_cfg["tf_dir"])/data_name/f"{encoded_run_name}/client-{d_identity}"

    if ('SLURM_JOBID' in os.environ) and ("tf_dir" in executor_cfg):
        executor_cfg["tf_dir"] = Path.home()/executor_cfg["tf_dir"]/data_name

    strategy = MasterWorkerStrategyV2(
        n_workers=n_workers-1,
        config=cfg,
        role=role
    )

    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        strategy=strategy,
        config=cfg
    )

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
