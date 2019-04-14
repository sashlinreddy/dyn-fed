import click
import os
import time 
import numpy as np

from fault_tolerant_ml.distribute import Master
from fault_tolerant_ml.distribute import MasterStrategy
from fault_tolerant_ml.ml.linear_model import LogisticRegression
from fault_tolerant_ml.ml.optimizer import SGDOptimizer
from fault_tolerant_ml.ml.loss_fns import cross_entropy_loss, cross_entropy_gradient
from fault_tolerant_ml.distribute.wrappers import ftml_train_collect, ftml_trainv2
from fault_tolerant_ml.data import MNist, OccupancyData
from fault_tolerant_ml.utils import setup_logger

# @ftml_trainv2
# def train_iter(master, *args, **kwargs):
#     # Map tasks
#     master.map()

#     # Gather and apply gradients
#     master.train_iteration(events)
#     train_gather(master)

# @ftml_train_collect
# def train_gather(master, *args, **kwargs):
#     theta_p = master.theta.copy()
#     # Receive updated parameters from workers
#     d_theta, epoch_loss = master.gather(events)

#     # Update the global parameters with weighted error
#     master.theta = master.optimizer.minimize(X=None, y=None, y_pred=None, theta=master.theta, precomputed_gradients=d_theta)

#     if master.tf_logger is not None:
#         master.tf_logger.histogram("theta-master", master.theta, master.dist_strategy.model.iter, bins=master.n_iterations)
#         master.tf_logger.scalar("epoch-master", epoch_loss, master.dist_strategy.model.iter)

#     delta = np.max(np.abs(theta_p - master.theta))

#     # self.logger.info(f"iteration = {self.dist_strategy.model.iter}, delta = {delta:7.4f}, Loss = {epoch_loss:7.4f}")

#     return d_theta, epoch_loss, delta

@click.command()
@click.option('--n_iterations', '-i', default=200, type=int)
@click.option('--learning_rate', '-lr', default=0.99, type=float)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--strategy', '-st', default="mw", type=str)
@click.option('--scenario', '-s', default=5, type=int)
@click.option('--remap', '-r', default=1, type=int)
@click.option('--quantize', '-q', default=0, type=int)
@click.option('--n_most_rep', '-nmr', default=100, type=int)
@click.option('--comm_period', '-cp', default=5, type=int)
@click.option('--clip_norm', '-cn', default=None, type=int)
@click.option('--clip_val', '-ct', default=None, type=int)
@click.option('--delta_switch', '-ds', default=1e-4, type=float)
@click.option('--shuffle', '-sh', default=1, type=int)
def run(n_iterations, learning_rate, verbose, strategy, scenario, remap, quantize, n_most_rep, comm_period, clip_norm, clip_val, delta_switch, shuffle):
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
        scenario (int): The scenario we would like to run. Default 1: Normal run, Scenario 2: Kill worker. Scenario 3: Kill worker and reintroduce another worker.
        n_most_rep (int): No. of most representative data points to keep track of for redistributing
        comm_period (int): Communicate parameters back to master every so often depending on this number
        delta_switch (float): Delta threshold to let us know when we switch back to communicating every ith iteration
        shuffle (bool): Flag whether or not to shuffle training data at each iteration
    """

    # # load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    loss = cross_entropy_loss
    grad = cross_entropy_gradient
    optimizer = SGDOptimizer(loss=loss, grad=grad, learning_rate=learning_rate, role="master", n_most_rep=n_most_rep, clip_norm=clip_norm, clip_val=clip_val)
    model = LogisticRegression(optimizer, max_iter=n_iterations, shuffle=shuffle)
    data_dir = "/c/Users/nb304836/Documents/git-repos/fault_tolerant_ml/data/"
    filepaths = {
        "train": {
            "images": os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        },
        "test": {
            "images": os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "labels": os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
        }
    }
    mnist = MNist(filepaths)
    logger = setup_logger(level=verbose)

    # For reproducibility
    np.random.seed(42)

    # data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
    # data.transform()

    dist_strategy = MasterStrategy(
        strategy=strategy,
        scenario=scenario,
        model=model,
        remap=remap,
        quantize=quantize,
        n_most_rep=n_most_rep, 
        comm_period=comm_period,
        delta_switch=delta_switch
    )

    master = Master(
        dist_strategy=dist_strategy,
        verbose=verbose,
    )

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

    # Plot class balances
    if "FIGDIR" in os.environ:

        import pandas as pd
        from fault_tolerant_ml.viz.target import ClassBalance

        figdir = os.environ["FIGDIR"]

        try:
            logger.debug("Saving class balances distribution plot...")
            worker_ids = [s.identity.decode() for s in master.watch_dog.states if s.state]
            fname = os.path.join(figdir, f"mnist-class-balance.png")
            class_bal = [v[1] for (k, v) in master.distributor.labels_per_worker.items() if k.identity.decode() in worker_ids]
            class_names = master.data.class_names

            class_balance = ClassBalance(labels=worker_ids, legend=class_names, fname=fname, stacked=True, percentage=True)
            class_balance.fit(y=class_bal)
            class_balance.poof()
        except Exception as e:
            logger.exception(e)

    logger.info("DONE!")

if __name__ == "__main__":
    run()
