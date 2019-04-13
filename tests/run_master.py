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
@click.option('--scenario', '-s', default=2, type=int)
@click.option('--n_most_rep', '-nmr', default=100, type=int)
@click.option('--comm_period', '-cp', default=10, type=int)
@click.option('--delta_switch', '-ds', default=0.0074, type=float)
def run(n_iterations, learning_rate, verbose, strategy, scenario, n_most_rep, comm_period, 
delta_switch):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training
        learning_rate (float): The rate at which we want our model to learn
        verbose (int): The logger level as an integer. See more in the logging file for different options
        scenario (int): The scenario we would like to run. Default 0: Redistribute all data points. Scenario 1: Quantize        parameters. Scenario 2: Redistribute data points of dead worker only
        n_most_rep (int): No. of most representative data points to keep track of for redistributing
        comm_period (int): Communicate parameters back to master every so often depending on this number
        delta_switch (float): Delta threshold to let us know when we switch back to communicating every ith iteration
    """

    # # load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    loss = cross_entropy_loss
    grad = cross_entropy_gradient
    optimizer = SGDOptimizer(loss=loss, grad=grad, learning_rate=learning_rate, role="master", n_most_rep=n_most_rep)
    model = LogisticRegression(optimizer, max_iter=n_iterations)
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

    # For reproducibility
    np.random.seed(42)

    # data = OccupancyData(filepath="/c/Users/nb304836/Documents/git-repos/large_scale_ml/data/occupancy_data/datatraining.txt", n_stacks=100)
    # data.transform()

    dist_strategy = MasterStrategy(
        strategy=strategy,
        scenario=scenario,
        n_most_rep=n_most_rep, 
        comm_period=comm_period,
        delta_switch=delta_switch,
        model=model
    )

    master = Master(
        dist_strategy=dist_strategy,
        verbose=verbose,
    )
    master.connect()
    # setattr(master, "train_iter", train_iter)
    time.sleep(1)

    master.train(mnist)

if __name__ == "__main__":
    run()
