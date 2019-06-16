import click
import os
import time

from fault_tolerant_ml.distribute import Worker
from fault_tolerant_ml.distribute import MasterWorkerStrategy
from fault_tolerant_ml.lib.io import file_io
from fault_tolerant_ml.ml.linear_model import LogisticRegression
from fault_tolerant_ml.ml.optimizer import SGDOptimizer
from fault_tolerant_ml.ml import loss_fns

@click.command()
@click.argument('n_workers', type=int)
@click.option('--verbose', '-v', default=10, type=int)
@click.option('--id', '-i', default="", type=str)
@click.option('--tmux', '-t', default=0, type=int)
@click.option('--add', '-a', default=0, type=int)
def run(n_workers, verbose, id, tmux, add):
    """Run worker

    Args:
        verbose (int): The debug level for the logging module
    """
    # load_dotenv(find_dotenv())

    # Create worker identity
    identity: int = 0

    if tmux:
        identity = id=int(id[1:]) if id != "" else None
    else:
        identity = int(id) if id != "" else None
        if add:
            identity += 1000

    # Load model config - this is an assumption that config is in root code directory
    cfg = file_io.load_model_config('config.yml')

    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
    executor_cfg = cfg['executor']

    
    # Setup optimizer
    gradient = loss_fns.cross_entropy_gradient
    optimizer = SGDOptimizer(
        loss=loss_fns.single_cross_entropy_loss, 
        grad=gradient, 
        role="worker", 
        learning_rate=opt_cfg['learning_rate'], 
        n_most_rep=opt_cfg['n_most_rep'], 
        clip_norm=None,
        mu_g=opt_cfg['mu_g']
    )

    # Create model
    model = LogisticRegression(optimizer, max_iter=model_cfg['n_iterations'], shuffle=model_cfg['shuffle'])

    # Setup distribution strategy
    strategy = MasterWorkerStrategy(
        model=model,
        n_workers=n_workers,
        config=executor_cfg
    )

    worker = Worker(
        strategy=strategy,
        n_workers=n_workers,
        verbose=verbose,
        id=identity
    )

    time.sleep(1)

    worker.connect()
    # time.sleep(1)
    worker.start()

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter