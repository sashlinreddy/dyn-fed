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
    def __init__(self, strategy, scenario, model, data_path):
        self.strategy = strategy
        self.scenario = scenario
        self.model = model
        self.data_path = data_path

class MasterStrategy(DistributionStrategy):

    def __init__(self, strategy, scenario, n_most_rep, comm_period, delta_switch, model, data_path):
        super().__init__(strategy, scenario, model, data_path)
        self.n_most_rep = n_most_rep
        self.comm_period = comm_period
        self.delta_switch = delta_switch