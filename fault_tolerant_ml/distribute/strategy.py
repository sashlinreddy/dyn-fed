from fault_tolerant_ml.distribute import Master

class DistributionStrategy(object):
    """Strategy for distributing within the cluster
    
    Responsible for master-worker strategy. Contains all variables regarding the strategy
    
    Attributes:
        type (int): strategy type
        n_most_rep (int): No. of most representative points
        comm_period (int): Communicate every comm_period iteration(s)
        delta_switch (float): The delta threshold we reach before switching back to communication every iteration
    """
    def __init__(self, config):
        assert config
        self.strategy = config['strategy']
        self.scenario = config['scenario']

    @property
    def name(self):
        raise NotImplementedError("Child should override this")

class LocalStrategy(DistributionStrategy):
    
    def __init__(self, config):
        pass

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
        aggregate_mode (int): Type of aggregation to be used (weighted average , weighted by loss, etc)
        delta_switch (float): The delta threshold we reach before switching back to communication every iteration
        worker_timeout (int): Timeout to pickup workers
        send_gradients (int): Whether or not to send gradients or parameters
        shared_folder (str): Path to shared data folder
        role (str): Master or worker role
    """
    def __init__(self, n_workers, config, role='master'):

        super().__init__(config)
        self.n_workers = n_workers
        self.remap = config['remap']
        self.quantize = config['quantize']
        self.comm_period = config['comm_period']
        self.aggregate_mode = config['aggregate_mode']
        self.delta_switch = config['delta_switch']
        self.worker_timeout = config['timeout']
        self.send_gradients = config['send_gradients']
        self.shared_folder = config['shared_folder']
        self.role = role
        if self.role =='worker':
            self.identity = config['identity']

    @property
    def name(self):
        return "master_worker"