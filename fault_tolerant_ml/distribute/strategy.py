"""Strategy for distribution
"""

class DistributionStrategy():
    """Strategy for distributing within the cluster
    
    Responsible for master-worker strategy. Contains all variables regarding the strategy
    
    Attributes:
        type (int): strategy type
        n_most_rep (int): No. of most representative points
        comm_period (int): Communicate every comm_period iteration(s)
        delta_switch (float): The delta threshold we reach before switching back
        to communication every iteration
    """
    def __init__(self, config):
        self.strategy = config.get('strategy')
        self.scenario = config.get('scenario')

    @property
    def name(self):
        """Returns name of strategy
        """
        raise NotImplementedError("Child should override this")

class LocalStrategy(DistributionStrategy):
    """Local strategy
    """
    def __init__(self, config):
        super().__init__(config)

    @property
    def name(self):
        return "local"

class MasterWorkerStrategy(DistributionStrategy):
    """Master worker strategy for distributing within the cluster
    
    Attributes:
        n_workers (int): No. of workers being used for session
        remap (int): Remap strategy
        quantize (int): Whether or not to quantize parameters
        comm_period (int): Communicate every comm_period iteration(s)
        aggregate_mode (int): Type of aggregation to be used (weighted average,
        weighted by loss, etc)
        delta_switch (float): The delta threshold we reach before switching back
        to communication every iteration
        worker_timeout (int): Timeout to pickup workers
        send_gradients (int): Whether or not to send gradients or parameters
        shared_folder (str): Path to shared data folder
        role (str): Master or worker role
    """
    def __init__(self, n_workers, config, role='master'):
        super().__init__(config)
        self.n_workers = n_workers
        self.comm_period = config.get('interval')
        self.comm_mode = config.get('mode')

        self.remap = config.get('remap')
        self.quantize = config.get('quantize')
        self.overlap = config.get("overlap")
        self.aggregate_mode = config.get('aggregate_mode')
        self.delta_switch = config.get('delta_switch')
        self.worker_timeout = config.get('timeout')
        self.send_gradients = config.get('send_gradients')

        self.shared_folder = config.get('shared_folder')
        self.config_folder = config.get("config_folder")
        
        self.role = role
        if self.role == 'worker':
            self.identity = config['identity']

    @property
    def name(self):
        return "master_worker"
