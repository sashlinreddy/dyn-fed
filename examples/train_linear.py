"""Linear regression example
"""

import logging
import os

import click
import numpy as np
# For reproducibility
# pylint: disable=wrong-import-position
np.random.seed(42)
from dotenv import find_dotenv, load_dotenv

from dyn_fed.data import MNist
from dyn_fed.distribute import MasterWorkerStrategy
from dyn_fed.lib.io import file_io
from dyn_fed.losses import MSELoss
from dyn_fed.metrics import accuracy_scorev2, confusion_matrix
from dyn_fed.optimizers import SGD, Adam
from dyn_fed.utils import model_utils, setup_logger
from ft_models import LinearRegression


@click.command()
@click.argument('n_workers', type=int)
@click.option('--role', '-r', default="client", type=str)
@click.option('--verbose', '-v', default="INFO", type=str)
@click.option('--identity', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
@click.option('--add', '-a', default=0, type=int)
def run(n_workers, role, verbose, identity, tmux, add):
    """Controller function which creates the server and starts off the training

    Args:
        n_workers (int): No. of clients to be used for the session
        verbose (int): The logger level as an integer. See more
        in the logging file for different options
    """
    if not "SLURM_JOBID" in os.environ:
        load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        pass # Not doing anything for now in terms of flushing directories
        # from dyn_fed.lib.io.file_io import flush_dir
        # _ = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        # flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    # Load in config to setup model
    config_path = 'config/config.yml'
    if 'PROJECT_DIR' in os.environ:
        config_path = os.path.join(os.environ['PROJECT_DIR'], config_path)
        
    cfg = file_io.load_yaml(config_path)

    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
    executor_cfg = cfg['executor']

    # Create identity
    d_identity: int = 0

    if tmux:
        d_identity = int(identity[1:]) if identity != "" else None
    else:
        d_identity = int(identity) if identity != "" else None
        if add:
            d_identity += 1000

    executor_cfg['identity'] = d_identity

    if 'PROJECT_DIR' in os.environ:
        executor_cfg['shared_folder'] = os.path.join(
            os.environ['PROJECT_DIR'], 
            executor_cfg['shared_folder']
        )
        executor_cfg['config_folder'] = os.path.join(
            os.environ['PROJECT_DIR'], 
            executor_cfg['config_folder']
        )

    # Encode run name for logs
    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)

    logger = None
    data = None

    if role == "server":
        # Setup logger
        setup_logger(level=verbose)

        # Master reads in data
        data_dir = executor_cfg['shared_folder']

        # Get data
        filepaths = {
            "train": {
                "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
            },
            "test": {
                "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
            }
        }
        data = MNist(filepaths)

        # data = OccupancyData(filepath="data/occupancy_data/datatraining.txt", n_stacks=100)
        # data.transform()    

    # Define loss
    loss = MSELoss()

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

    model = LinearRegression(
        optimizer,
        strategy,
        batch_size=model_cfg["batch_size"],
        max_iter=model_cfg['n_iterations'],
        shuffle=model_cfg['shuffle'],
        verbose=verbose,
        encode_name=encoded_run_name
    )

    try:
        logger = logging.getLogger("dfl.scripts.train")
        logger.info(f"Starting run: {encoded_run_name}")
        logger.info(f"Optimizer={optimizer}")

        logger.info("*******************************")
        logger.info("STARTING TRAINING")
        logger.info("*******************************")

        # Learn model parameters
        model.fit(data)

        logger.info("*******************************")
        logger.info("COMPLETED TRAINING")
        logger.info("*******************************")

        if role == "server":
            
            # Print confusion matrix
            y_pred = model.forward(data.X_test)
            # conf_matrix = confusion_matrix(self.data.y_test, y_pred)
            conf_matrix = confusion_matrix(data.y_test.data, y_pred.data)
            logger.info(f"Confusion matrix=\n{conf_matrix}")

            # Accuracy
            # acc = accuracy_scorev2(self.data.y_test, y_pred)
            acc = accuracy_scorev2(data.y_test.data, y_pred.data)
            logger.info(f"Accuracy={acc * 100:7.4f}%")

            # Plot metrics
            # model.plot_metrics()

    except Exception as e:
        logger.exception(e)
        raise
    finally:
        logger.info("DONE!")
        logging.shutdown()

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
