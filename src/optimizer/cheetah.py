from src.simulator.domain import Domain

class CheetahOptimizer:
    def __init__(self, domain: Domain):
        self.domain = domain
    
    def do(self):
        raise NotImplementedError("main finction to run algorithm, not iemented")