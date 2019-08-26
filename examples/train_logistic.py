import click
import os
import time 
import numpy as np
import logging
from dotenv import load_dotenv, find_dotenv

from fault_tolerant_ml.distribute import MasterWorkerStrategy
from fault_tolerant_ml.distribute.wrappers import ftml_train_collect, ftml_trainv2
from fault_tolerant_ml.data import MNist, OccupancyData
from fault_tolerant_ml.losses import cross_entropy_loss, cross_entropy_gradient, CrossEntropyLoss
from fault_tolerant_ml.optimizers import SGDOptimizer, AdamOptimizer
from fault_tolerant_ml.models.linear_model import LogisticRegression
from fault_tolerant_ml.metrics import confusion_matrix, accuracy_scorev2
from fault_tolerant_ml.utils import setup_logger, model_utils
from fault_tolerant_ml.lib.io import file_io

@click.command()
@click.argument('n_workers', type=int)
@click.option('--role', '-r', default="worker", type=str)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--id', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
@click.option('--add', '-a', default=0, type=int)
def run(n_workers, role, verbose, id, tmux, add):
    """Controller function which creates the master and starts off the training

    Args:
        n_workers (int): No. of workers to be used for the session
        verbose (int): The logger level as an integer. See more in the logging file for different options

    """

    load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        _ = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        # flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    # Load in config to setup model
    config_path = 'config.yml'
    if 'PROJECT_DIR' in os.environ:
        config_path = os.path.join(os.environ['PROJECT_DIR'], config_path)
        
    cfg = file_io.load_model_config(config_path)

    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
    executor_cfg = cfg['executor']

    # Create identity
    identity: int = 0

    if tmux:
        identity = id=int(id[1:]) if id != "" else None
    else:
        identity = int(id) if id != "" else None
        if add:
            identity += 1000

    executor_cfg['identity'] = identity

    if 'PROJECT_DIR' in os.environ:
        executor_cfg['shared_folder'] = os.path.join(os.environ['PROJECT_DIR'], executor_cfg['shared_folder'])

    # Encode run name for logs
    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)

    # For reproducibility
    np.random.seed(42)

    logger = None
    data = None

    if role == "master":
        # Setup logger
        setup_logger(level=verbose)

        # Master reads in data
        data_dir = executor_cfg['shared_folder']

        # Get data
        filepaths = {
            "train": {
                "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
            },
            "test": {
                "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
            }
        }
        data = MNist(filepaths)

        # data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
        # data.transform()    

        # time.sleep(2)

    # Define loss
    loss = CrossEntropyLoss()

    # Define optimizer
    optimizer = None

    if opt_cfg["name"] == "sgd":
        optimizer = SGDOptimizer(
            loss=loss, 
            learning_rate=opt_cfg['learning_rate'], 
            role=role, 
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )
    elif opt_cfg["name"] == "adam":
        optimizer = AdamOptimizer(
            loss=loss, 
            learning_rate=opt_cfg['learning_rate'],
            role=role,
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )

    # Decide on distribution strategy
    strategy = MasterWorkerStrategy(
        n_workers=n_workers,
        config=executor_cfg,
        role=role
    )

    # Create model
    model = LogisticRegression(
        optimizer, 
        strategy, 
        max_iter=model_cfg['n_iterations'], 
        shuffle=model_cfg['shuffle'], 
        verbose=verbose,
        encode_name=encoded_run_name
    )

    try:
        logger = logging.getLogger("ftml.scripts.train")
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

        if role == "master":
            
            # Print confusion matrix
            y_pred = model.predict(data.X_test)
            # conf_matrix = confusion_matrix(self.data.y_test, y_pred)
            conf_matrix = confusion_matrix(data.y_test.data, y_pred.data)
            logger.info(f"Confusion matrix=\n{conf_matrix}")

            # Accuracy
            # acc = accuracy_scorev2(self.data.y_test, y_pred)
            acc = accuracy_scorev2(data.y_test.data, y_pred.data)
            logger.info(f"Accuracy={acc * 100:7.4f}%")

            # Plot metrics
            model.plot_metrics()

    except Exception as e:
        logger.exception(e)
        raise
    finally:
        logger.info("DONE!")
        logging.shutdown()

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
