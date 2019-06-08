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
    def __init__(self, strategy, scenario, model, n_workers):
        self.strategy = strategy
        self.scenario = scenario
        self.model = model
        self.n_workers = n_workers

class MasterStrategy(DistributionStrategy):

    def __init__(
        self, 
        strategy, 
        scenario, 
        model,
        n_workers, 
        remap=0, 
        quantize=0, 
        n_most_rep=100, 
        comm_period=1, 
        delta_switch=1e-4, 
        worker_timeout=10,
        mu_g=1.0
        ):

        super().__init__(strategy, scenario, model, n_workers)
        self.remap = remap
        self.quantize = quantize
        self.n_most_rep = n_most_rep
        self.comm_period = comm_period
        self.delta_switch = delta_switch
        self.worker_timeout = worker_timeout
        self.mu_g = mu_g

    def encode(self):
        return f"{self.n_workers}-{self.scenario}-{self.remap}-{self.quantize}-{self.n_most_rep}-{self.comm_period}-{self.mu_g}"