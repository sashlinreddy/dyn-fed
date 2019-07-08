import click
import os
import time 
import numpy as np

from fault_tolerant_ml.distribute import MasterWorkerStrategy
from fault_tolerant_ml.ml.linear_model import LogisticRegression
from fault_tolerant_ml.ml.optimizer import SGDOptimizer, AdamOptimizer
from fault_tolerant_ml.ml.loss_fns import cross_entropy_loss, cross_entropy_gradient
from fault_tolerant_ml.distribute.wrappers import ftml_train_collect, ftml_trainv2
from fault_tolerant_ml.data import MNist, OccupancyData
from fault_tolerant_ml.utils import setup_logger, model_utils
from fault_tolerant_ml.lib.io import file_io

@click.command()
@click.argument('n_workers', type=int)
@click.option('--verbose', '-v', default=10, type=int)
def run(n_workers, verbose):
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
        scenario (int): The scenario we would like to run. Default 1: Normal run, Scenario 2: Kill worker. Scenario 3: Kill     
            worker and reintroduce another worker. Scenario 4: Communicate every 10 iterations, Scenario 5: Every 5             
            iterations and gradient clipping?
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
    config_path = 'config.yml'
    if 'PROJECT_DIR' in os.environ:
        config_path = os.path.join(os.environ['PROJECT_DIR'], config_path)
        
    cfg = file_io.load_model_config(config_path)
    
    model_cfg = cfg['model']
    opt_cfg = cfg['optimizer']
    executor_cfg = cfg['executor']

    data_dir = executor_cfg['shared_folder']
    if 'PROJECT_DIR' in os.environ:
        data_dir = os.path.join(os.environ['PROJECT_DIR'], data_dir)
        executor_cfg['shared_folder'] = data_dir

    # encode_vars = [
    #     "n_workers", "scenario", "remap", "quantize" , "n_most_rep",
    #     "comm_period", "mu_g", "send_gradients"
    # ]

    # global_cfg = {"n_workers": n_workers}
    # global_cfg.update(executor_cfg)
    # global_cfg.update(opt_cfg)
    # encode_name = string_utils.dict_to_str(global_cfg, choose=encode_vars)

    # if "LOGDIR" in os.environ:
    #     logdir = os.path.join(os.environ["LOGDIR"], encode_name)
    #     if not os.path.exists(logdir):
    #         try:
    #             os.mkdir(logdir)
    #         except FileExistsError:
    #             pass
    #     os.environ["LOGDIR"] = logdir

    encoded_run_name = model_utils.encode_run_name(n_workers, cfg)

    logger = setup_logger(level=verbose)

    logger.info(f"Starting run: {encoded_run_name}")

    # Create optimizer
    loss = cross_entropy_loss
    grad = cross_entropy_gradient
    # optimizer = SGDOptimizer(
    #     loss=loss, 
    #     grad=grad, 
    #     learning_rate=opt_cfg['learning_rate'], 
    #     role="master", 
    #     n_most_rep=opt_cfg['n_most_rep'], 
    #     mu_g=opt_cfg['mu_g']
    # )

    optimizer = AdamOptimizer(
        loss=loss, 
        grad=grad, 
        learning_rate=opt_cfg['learning_rate'],
        role="master",
        n_most_rep=opt_cfg['n_most_rep'], 
        mu_g=opt_cfg['mu_g']
    )

    # Decide on distribution strategy
    strategy = MasterWorkerStrategy(
        n_workers=n_workers,
        config=executor_cfg
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

    # Get data
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

    # time.sleep(2)

    logger.info("*******************************")
    logger.info("STARTING TRAINING")
    logger.info("*******************************")

    # master.train(mnist)
    model.fit(mnist)

    logger.info("*******************************")
    logger.info("COMPLETED TRAINING")
    logger.info("*******************************")

    model.plot_metrics()

    logger.info("DONE!")

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
