"""Logistic regression example
"""
import logging
import os
from pathlib import Path

import click
import numpy as np
# For reproducibility
# pylint: disable=wrong-import-position
np.random.seed(42)

from fault_tolerant_ml.data import MNist
from fault_tolerant_ml.distribute import MasterWorkerStrategy

from fault_tolerant_ml.lib.io import file_io
from fault_tolerant_ml.losses import CrossEntropyLoss
from fault_tolerant_ml.metrics import accuracy_scorev2, confusion_matrix
from fault_tolerant_ml.optimizers import SGD, Adam
from fault_tolerant_ml.utils import model_utils, setup_logger

from ft_models import LogisticRegressionV2

def train(data,
          role,
          n_workers,
          model_cfg,
          opt_cfg,
          executor_cfg,
          verbose,
          encoded_run_name):
    """Train model
    """
    # Define loss
    loss = CrossEntropyLoss()

    # Define optimizer
    optimizer = None

    if opt_cfg["name"] == "sgd":
        optimizer = SGD(
            loss=loss, 
            learning_rate=opt_cfg['learning_rate'], 
            role=role, 
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )
    elif opt_cfg["name"] == "adam":
        optimizer = Adam(
            loss=loss, 
            learning_rate=opt_cfg['learning_rate'], 
            role=role, 
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )

    # Decide on distribution strategy
    strategy = MasterWorkerStrategy(
        n_workers=n_workers-1,
        config=executor_cfg,
        role=role
    )

    # Create model
    model = LogisticRegressionV2(
        optimizer, 
        strategy, 
        max_iter=model_cfg['n_iterations'], 
        shuffle=model_cfg['shuffle'], 
        verbose=verbose,
        encode_name=encoded_run_name
    )

    logger = logging.getLogger("ftml.examples.train_logisticv2")
    
    try:
        logger.info(f"Starting run: {encoded_run_name}")
        logger.info(f"Optimizer={optimizer}")

        logger.info("*******************************")
        logger.info("STARTING TRAINING")
        logger.info("*******************************")

        # Learn model parameters
        if data:
            model.fit(data.X_train, data.y_train, data.X_test, data.y_test)
        else:
            model.fit()

        logger.info("*******************************")
        logger.info("COMPLETED TRAINING")
        logger.info("*******************************")

        if role == "master":
            
            # Print confusion matrix
            y_pred = model.forward(data.X_test)
            conf_matrix = confusion_matrix(data.y_test.data, y_pred.data)
            logger.info(f"Confusion matrix=\n{conf_matrix}")

            # Accuracy
            acc = accuracy_scorev2(data.y_test.data, y_pred.data)
            logger.info(f"Accuracy={acc * 100:7.4f}%")

            # Plot metrics
            # model.plot_metrics()
    
    except Exception as e:
        logger.exception(e)
        raise
    finally:
        logger.info("DONE!")
        logger.handlers = []
        logging.shutdown()


@click.command()
@click.argument('n_workers', type=int)
@click.option('--role', '-r', default="worker", type=str)
@click.option('--verbose', '-v', default="INFO", type=str)
@click.option('--identity', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
@click.option('--add', '-a', default=0, type=int)
@click.option('--config', '-c', default='config/config.yml', type=str)
def run(n_workers, role, verbose, identity, tmux, add, config):
    """Controller function which creates the master and starts off the training

    Args:
        n_workers (int): No. of workers to be used for the session
        verbose (int): The logger level as an integer. See more in the logging
            file for different options
    """
    if not "SLURM_JOBID" in os.environ:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        pass # Not doing anything for now in terms of flushing logs
        # from fault_tolerant_ml.lib.io.file_io import flush_dir
        # _ = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        # flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    # Load in config to setup model
    config_path = config
    if 'PROJECT_DIR' in os.environ:
        config_path = Path(os.environ['PROJECT_DIR'])/config_path
        
    cfg = file_io.load_model_config(config_path)

    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
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

    data = None

    if role == "master":
        # Setup logger
        setup_logger(level=verbose)

        # Master reads in data
        data_dir = Path(executor_cfg['shared_folder'])

        # Get data
        filepaths = {
            "train": {
                "images": data_dir/"train-images-idx3-ubyte.gz",
                "labels": data_dir/"train-labels-idx1-ubyte.gz"
            },
            "test": {
                "images": data_dir/"t10k-images-idx3-ubyte.gz",
                "labels": data_dir/"t10k-labels-idx1-ubyte.gz"
            }
        }
        data = MNist(filepaths, noniid=executor_cfg['noniid'])

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = Path(executor_cfg["tf_dir"])/f"{encoded_run_name}/master"

        # data = OccupancyData(
        #     filepath="/data/occupancy_data/datatraining.txt",
        #     n_stacks=100
        # )
        # data.transform()    

        # time.sleep(2)
    else:
        setup_logger(filename=f'log-worker-{d_identity}.log', level=verbose)

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = Path(executor_cfg["tf_dir"])/f"{encoded_run_name}/worker-{d_identity}"

    train(
        data,
        role,
        n_workers,
        model_cfg,
        opt_cfg,
        executor_cfg,
        verbose,
        encoded_run_name
    )

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
