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

    def __init__(self, n_workers, config, role='master'):

        super().__init__(config)
        self.n_workers = n_workers
        self.remap = config['remap']
        self.quantize = config['quantize']
        self.comm_period = config['comm_period']
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

    # def encode(self):
    #     return f"{self.n_workers}-{self.scenario}-{self.remap}-{self.quantize}-{self.n_most_rep}"
    #     f"-{self.comm_period}-{self.mu_g}-{self.send_gradients}"