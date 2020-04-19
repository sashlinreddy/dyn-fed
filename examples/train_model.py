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

from dyn_fed.data import OccupancyData
from dyn_fed.data import mnist, fashion_mnist
from dyn_fed.distribute import MasterWorkerStrategy

from dyn_fed.lib.io import file_io
from dyn_fed.losses import CrossEntropyLoss
from dyn_fed.metrics import accuracy_scorev2, confusion_matrix
from dyn_fed.optimizers import SGD, Adam
from dyn_fed.utils import model_utils, setup_logger

from ft_models import LogisticRegressionV2, SimpleNN, NN2

def train(data,
          role,
          n_workers,
          cfg,
          verbose,
          encoded_run_name):
    """Train model
    """
    # Define loss
    loss = CrossEntropyLoss()

    # Define optimizer
    optimizer = None

    if cfg.optimizer.name == "sgd":
        optimizer = SGD(
            loss=loss, 
            learning_rate=cfg.optimizer.learning_rate, 
            role=role, 
            n_most_rep=cfg.optimizer.n_most_rep, 
            mu_g=cfg.optimizer.mu_g
        )
    elif cfg.optimizer.name == "adam":
        optimizer = Adam(
            loss=loss, 
            learning_rate=cfg.optimizer.learning_rate, 
            role=role, 
            n_most_rep=cfg.optimizer.n_most_rep, 
            mu_g=cfg.optimizer.mu_g
        )

    # Decide on distribution strategy
    strategy = MasterWorkerStrategy(
        n_workers=n_workers-1,
        config=cfg,
        role=role
    )

    # Create model
    if cfg.model.type == "logistic":
        model = LogisticRegressionV2(
            optimizer, 
            strategy,
            batch_size=cfg.data.batch_size,
            max_iter=cfg.model.n_iterations, 
            shuffle=cfg.data.shuffle, 
            verbose=verbose,
            encode_name=encoded_run_name
        )
    elif cfg.model.type == "nn1":
        model = SimpleNN(
            optimizer, 
            strategy,
            batch_size=cfg.data.batch_size,
            max_iter=cfg.model.n_iterations, 
            shuffle=cfg.data.shuffle, 
            verbose=verbose,
            encode_name=encoded_run_name
        )
    elif cfg.model.type == "nn2":
        model = NN2(
            optimizer, 
            strategy,
            batch_size=cfg.data.batch_size,
            max_iter=cfg.model.n_iterations, 
            shuffle=cfg.data.shuffle, 
            verbose=verbose,
            encode_name=encoded_run_name
        )

    logger = logging.getLogger("dfl.examples.train_logisticv2")
    
    try:
        logger.info(f"Starting run: {encoded_run_name}")
        logger.info(f"Optimizer={optimizer}")

        logger.info("*******************************")
        logger.info("STARTING TRAINING")
        logger.info("*******************************")

        if isinstance(data, tuple):
            if role == "server":
                X_train, y_train, X_test, y_test = data
            else:
                X_test, y_test = data # Unpack tuple
        else:
            if role == "server": # Old data class
                X_train, y_train, X_test, y_test = (
                    data.X_train,
                    data.y_train,
                    data.X_test,
                    data.y_test
                )

        # Learn model parameters
        if role == "server":
            model.fit(X_train, y_train, X_test, y_test)
        else:
            # Clients will only have test set if we are checking overfitting
            if cfg.model.check_overfitting:
                model.fit(X_valid=X_test, y_valid=y_test)
            else:
                model.fit()

        logger.info("*******************************")
        logger.info("COMPLETED TRAINING")
        logger.info("*******************************")

        if role == "server":
            
            # Print confusion matrix
            y_pred = model.forward(X_test)
            conf_matrix = confusion_matrix(y_test.data, y_pred.data)
            logger.info(f"Confusion matrix=\n{conf_matrix}")

            # Accuracy
            acc = accuracy_scorev2(y_test.data, y_pred.data)
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

    executor_cfg = cfg.executor

    executor_cfg.update(cfg.distribute)
    executor_cfg.update(cfg.comms)

    # Create identity
    d_identity: int = 0

    if tmux:
        d_identity = identity = int(identity[1:]) if identity != "" else None
    else:
        d_identity = int(identity) if identity != "" else None
        if add:
            d_identity += 1000

    executor_cfg.identity = d_identity

    if 'PROJECT_DIR' in os.environ:
        executor_cfg['shared_folder'] = (
            Path(os.environ['PROJECT_DIR'])/executor_cfg['shared_folder']
        )
        executor_cfg['config_folder'] = (
            Path(os.environ['PROJECT_DIR'])/executor_cfg['config_folder']
        )

    # Encode run name for logs
    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)

    # Validate config
    if cfg.comms.interval > cfg.model.n_iterations:
        cfg.comms.interval = cfg.model.n_iterations

    data = None

    data_name = Path(cfg.data.name)
    opt_name = Path(cfg.optimizer.name)
    model_type = Path(cfg.model.type)

    if role == "server":
        # Setup logger
        logger = setup_logger(level=verbose)

        slurm_jobid = os.environ.get("SLURM_JOBID", None)

        if slurm_jobid is not None:
            logger.info(f"SLURM_JOBID={slurm_jobid}")

        # Server reads in data
        if str(data_name) == "fashion-mnist":
            logger.info("Dataset: Fashion-MNist")
            data = fashion_mnist.load_data(
                noniid=cfg.data.noniid,
                onehot=True, # One hot encode labels
                convert_types=True, # Converts labels to 0.01 and 0.99 / to float 32
                reshape=True, # Reshape to 2D
                tensorize=True
            )
        elif str(data_name) == "mnist":
            logger.info("Dataset: MNist")
            data = mnist.load_data(
                noniid=cfg.data.noniid,
                onehot=True, # One hot encode labels
                convert_types=True, # Converts labels to 0.01 and 0.99 / to float 32
                reshape=True, # Reshape to 2D
                tensorize=True
            )

        elif str(data_name) == "occupancy_data":
            logger.info("Dataset: Occupancy data")
            data = OccupancyData(
                filepath="data/occupancy_data/datatraining.txt",
                n_stacks=100
            )
        else:
            raise Exception("Please enter valid dataset")

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = (
                Path(
                    executor_cfg["tf_dir"]
                )/data_name/model_type/opt_name/f"{encoded_run_name}/server"
            )    

        # time.sleep(2)
    else:
        logger = setup_logger(filename=f'log-client-{d_identity}.log', level=verbose)

        if cfg.model.check_overfitting:
            if cfg.data.name == "mnist":
                logger.info("Dataset: MNist, test set only")
                data = mnist.load_data(
                    noniid=cfg.data.noniid,
                    onehot=True, # One hot encode labels
                    convert_types=True, # Converts labels to 0.01 and 0.99 / to float 32
                    reshape=True, # Reshape to 2D
                    test_only=True, # Load only test set
                    tensorize=True # Convert to our special tensor
                )
                logger.info(f"Loaded in test set with shape={data[0].shape}")
            elif cfg.data.name == "fashion-mnist":
                logger.info("Dataset: Fashion-MNist, test set only")
                data = fashion_mnist.load_data(
                    noniid=cfg.data.noniid,
                    onehot=True, # One hot encode labels
                    convert_types=True, # Converts labels to 0.01 and 0.99 / to float 32
                    reshape=True, # Reshape to 2D
                    test_only=True, # Load only test set
                    tensorize=True # Convert to our special tensor
                )
                logger.info(f"Loaded in test set with shape={data[0].shape}")

        if "tf_dir" in executor_cfg:
            executor_cfg["tf_dir"] = Path(
                executor_cfg["tf_dir"]
            )/data_name/model_type/opt_name/f"{encoded_run_name}/client-{d_identity}"

    if ('SLURM_JOBID' in os.environ) and ("tf_dir" in executor_cfg):
        executor_cfg["tf_dir"] = (
            Path.home()/executor_cfg["tf_dir"]
        )

    train(
        data,
        role,
        n_workers,
        cfg,
        verbose,
        encoded_run_name
    )

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
