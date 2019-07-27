import click
import os
import time
from dotenv import load_dotenv, find_dotenv

from fault_tolerant_ml.distribute import MasterWorkerStrategy
from fault_tolerant_ml.lib.io import file_io
from fault_tolerant_ml.models.linear_model import LogisticRegression
from fault_tolerant_ml.optimizers import SGDOptimizer, AdamOptimizer
from fault_tolerant_ml import losses as loss_fns
from fault_tolerant_ml.utils import model_utils

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

     # Load in config to setup model
    config_path = 'config.yml'
    if 'PROJECT_DIR' in os.environ:
        config_path = os.path.join(os.environ['PROJECT_DIR'], config_path)
        
    cfg = file_io.load_model_config(config_path)

    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
    executor_cfg = cfg['executor']
    executor_cfg['identity'] = identity

    if 'PROJECT_DIR' in os.environ:
        executor_cfg['shared_folder'] = os.path.join(os.environ['PROJECT_DIR'], executor_cfg['shared_folder'])

    # Encode run name for logs
    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)
    
    # Setup optimizer
    gradient = loss_fns.cross_entropy_gradient

    if opt_cfg["name"] == "sgd":
        optimizer = SGDOptimizer(
            loss=loss_fns.single_cross_entropy_loss, 
            grad=gradient, 
            role="worker", 
            learning_rate=opt_cfg['learning_rate'], 
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )
    elif opt_cfg["name"] == "adam":
        optimizer = AdamOptimizer(
            loss=loss_fns.single_cross_entropy_loss, 
            grad=gradient, 
            role="worker", 
            learning_rate=opt_cfg['learning_rate'], 
            n_most_rep=opt_cfg['n_most_rep'], 
            mu_g=opt_cfg['mu_g']
        )

    # Setup distribution strategy
    strategy = MasterWorkerStrategy(
        n_workers=n_workers,
        config=executor_cfg,
        role='worker'
    )

    # Create model
    model = LogisticRegression(
        optimizer, 
        strategy, 
        max_iter=model_cfg['n_iterations'], 
        shuffle=model_cfg['shuffle'], 
        verbose=verbose,
        encode_name=encoded_run_name)
    
    # time.sleep(1)
    # Train model
    model.fit()

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter