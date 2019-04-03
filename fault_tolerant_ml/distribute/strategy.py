class DistributionStrategy(object):
    
    def __init__(self, type, n_most_rep, delay, delta_switch):
        self.type = type
        self.n_most_rep = n_most_rep
        self.delay = delay
        self.delta_switch = delta_switch