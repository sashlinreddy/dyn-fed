import click
import os
import time 
import numpy as np
import socket
import json

from fault_tolerant_ml.distribute import Master
from fault_tolerant_ml.distribute import MasterStrategy
from fault_tolerant_ml.ml.linear_model import LogisticRegression
from fault_tolerant_ml.ml.optimizer import SGDOptimizer
from fault_tolerant_ml.ml.loss_fns import cross_entropy_loss, cross_entropy_gradient
from fault_tolerant_ml.distribute.wrappers import ftml_train_collect, ftml_trainv2
from fault_tolerant_ml.data import MNist, OccupancyData
from fault_tolerant_ml.utils import setup_logger
from fault_tolerant_ml.lib.io import file_io

@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--verbose', '-v', default=10, type=int)
def run(data_dir, verbose):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training
        learning_rate (float): The rate at which we want our model to learn
        verbose (int): The logger level as an integer. See more in the logging file for different options
        remap (int): Type of remap strategy we use when a worker dies (default: 0).
            0: Redistribute all data points to alive workers
            1: Redistribute only most representative data points of dead worker to alive workers
        quantize (int): Whether or not to quantize parameters
            0: No quantization
            1: Min theta value, max theta value, interval, bins sent across the network to reconstruct the parameters on the worker side
        scenario (int): The scenario we would like to run. Default 1: Normal run, Scenario 2: Kill worker. Scenario 3: Kill     worker and reintroduce another worker. Scenario 4: Communicate every 10 iterations, Scenario 5: Every 5             iterations and gradient clipping?
        n_most_rep (int): No. of most representative data points to keep track of for redistributing
        comm_period (int): Communicate parameters back to master every so often depending on this number
        delta_switch (float): Delta threshold to let us know when we switch back to communicating every ith iteration
        shuffle (bool): Flag whether or not to shuffle training data at each iteration
        timeout (int): Time given for workers to connect to master
        send_gradients (bool): Whether or not to send gradients or parameters back
    """

    # # load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        # flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    # Load model config
    cfg = file_io.load_model_config('config.yml')['model']

    loss = cross_entropy_loss
    grad = cross_entropy_gradient
    optimizer = SGDOptimizer(
        loss=loss, 
        grad=grad, 
        learning_rate=cfg['learning_rate'], 
        role="master", 
        n_most_rep=cfg['n_most_rep'], 
        clip_norm=cfg['clip_norm'], 
        clip_val=cfg['clip_val'],
        mu_g=cfg['mu_g']
    )
    model = LogisticRegression(optimizer, max_iter=cfg['n_iterations'], shuffle=cfg['shuffle'])
    filepaths = {
        "train": {
            "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        },
        "test": {
            "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
        }
    }
    mnist = MNist(filepaths)

    # For reproducibility
    np.random.seed(42)

    # data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
    # data.transform()

    dist_strategy = MasterStrategy(
        n_workers=cfg['n_workers'],
        strategy=cfg['strategy'],
        scenario=cfg['scenario'],
        model=model,
        remap=cfg['remap'],
        quantize=cfg['quantize'],
        n_most_rep=cfg['n_most_rep'], 
        comm_period=cfg['comm_period'],
        delta_switch=cfg['delta_switch'],
        worker_timeout=cfg['timeout'],
        mu_g=cfg['mu_g'],
        send_gradients=cfg['send_gradients']
    )

    if "LOGDIR" in os.environ:
        logdir = os.path.join(os.environ["LOGDIR"], dist_strategy.encode())
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        os.environ["LOGDIR"] = logdir
        
    logger = setup_logger(level=verbose)

    master = Master(
        dist_strategy=dist_strategy,
        verbose=verbose,
    )

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    ip_config = {"ipAddress" : ip_address}

    logger.info(f"Master on ip={ip_address}")

    with open(os.path.join(data_dir, "ip_config.json"), "w") as f:
        json.dump(ip_config, f)

    # time.sleep(2)

    logger.info("Connecting master sockets")
    master.connect()
    # setattr(master, "train_iter", train_iter)
    time.sleep(1)

    logger.info("*******************************")
    logger.info("STARTING TRAINING")
    logger.info("*******************************")

    master.train(mnist)

    logger.info("*******************************")
    logger.info("COMPLETED TRAINING")
    logger.info("*******************************")

    master.plot_metrics()

    logger.info("DONE!")

if __name__ == "__main__":
    run()
