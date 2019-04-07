import click
import os
import time 

from fault_tolerant_ml.distribute import Master
from fault_tolerant_ml.distribute import MasterStrategy
from fault_tolerant_ml.ml.linear_model import LogisticRegression
from fault_tolerant_ml.ml.optimizer import SGDOptimizer
from fault_tolerant_ml.ml.loss_fns import cross_entropy_loss, cross_entropy_gradient

# @Master.ftml_train
def train_iteration(master):
    """Logic for one iteration of training
    """
    theta_p = master.model.theta
        
    # Map out data
    master.map()

    # Collect data
    d_theta, epoch_loss = master.collect()

    # Apply gradients to parameters
    master.model.iterate()

    delta = np.max(np.abs(theta_p - master.model.theta))

@click.command()
@click.option('--n_iterations', '-i', default=400, type=int)
@click.option('--learning_rate', '-lr', default=0.1, type=float)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--strategy', '-st', default="mw", type=str)
@click.option('--scenario', '-sc', default=0, type=int)
@click.option('--n_most_representative', '-nmr', default=100, type=int)
@click.option('--comm_period', '-cp', default=1, type=int)
@click.option('--delta_switch', '-ds', default=0.0074, type=float)
def run(n_iterations, learning_rate, verbose, strategy, scenario, n_most_representative, comm_period, 
delta_switch):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training (default: 400)
        learning_rate (float): The rate at which we want our model to learn (default: 0.1)
        verbose (int): The logger level as an integer. See more in the logging file for different options (default: 10)
        strategy (int): The scenario we would like to run (default: 1)
        n_most_representative (int): No. of most representative data points we will use when a worker dies (default: 100)
        comm_period (int): Communication period. We will communicate every so often. (default: 1)
        delta_switch (float): The delta threshold we reach before switching back to communication every iteration (default: 0.0074)
    """

    # # load_dotenv(find_dotenv())

    if "LOGDIR" in os.environ:
        from fault_tolerant_ml.lib.io.file_io import flush_dir
        ignore_dir = [os.path.join(os.environ["LOGDIR"], "tf/")]
        # ignore_dir = []
        flush_dir(os.environ["LOGDIR"], ignore_dir=ignore_dir)

    loss = cross_entropy_loss
    grad = cross_entropy_gradient
    optimizer = SGDOptimizer(loss=loss, grad=grad, learning_rate=learning_rate, role="master", n_most_rep=n_most_representative)
    model = LogisticRegression(optimizer, max_iter=n_iterations)
    data_path = "/c/Users/nb304836/Documents/git-repos/fault_tolerant_ml/data"

    dist_strategy = MasterStrategy(
        strategy=strategy,
        scenario=scenario,
        n_most_rep=n_most_representative, 
        comm_period=comm_period,
        delta_switch=delta_switch,
        model=model,
        data_path=data_path
    )

    master = Master(
        dist_strategy=dist_strategy,
        verbose=verbose
    )

    master.connect()
    time.sleep(1)
    master.start()

    # i = 0

    # while i < dist_strategy.model.max_iter:

    #     train_iteration()

@click.command()
@click.option('--n_iterations', '-i', default=400, type=int)
@click.option('--learning_rate', '-lr', default=0.1, type=float)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--strategy', '-st', default="mw", type=str)
@click.option('--scenario', '-s', default=0, type=int)
@click.option('--n_most_rep', '-nmr', default=100, type=int)
@click.option('--comm_period', '-cp', default=1, type=int)
@click.option('--delta_switch', '-ds', default=0.0074, type=float)
def run_old(n_iterations, learning_rate, verbose, strategy, scenario, n_most_rep, comm_period, 
delta_switch):
    """Controller function which creates the master and starts off the training

    Args:
        n_iterations (int): No. of iterations we perform for training
        learning_rate (float): The rate at which we want our model to learn
        verbose (int): The logger level as an integer. See more in the logging file for different options
        scenario (int): The scenario we would like to run
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
    data_path = "/c/Users/nb304836/Documents/git-repos/fault_tolerant_ml/data"

    dist_strategy = MasterStrategy(
        strategy=strategy,
        scenario=scenario,
        n_most_rep=n_most_rep, 
        comm_period=comm_period,
        delta_switch=delta_switch,
        model=model,
        data_path=data_path
    )

    master = Master(
        dist_strategy=dist_strategy,
        verbose=verbose,
    )
    master.connect()
    time.sleep(1)
    master.start()

if __name__ == "__main__":
    run_old()