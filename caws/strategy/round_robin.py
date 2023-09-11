from .base import Strategy

class FCFS_RoundRobin(Strategy):
    """ Simplest scheduling strategy made for testing and demonstration. Schedules tasks in a first 
    come first serve manner, distributes tasks  in a round-robin fashion across endpoints
    """

    def __init__(self, endpoints, transfer_predictor):
        super().__init__(endpoints, transfer_predictor)
        self.cur_idx = 0
        self.endpoints_list = endpoints
        self.n = len(self.endpoints_list)
    
    def schedule(self, tasks):
        schedule = [(t, self.endpoints_list[(self.cur_idx + i) % self.n]) for i,t in enumerate(tasks)]
        self.cur_idx = (self.cur_idx + len(tasks)) % self.n
        return schedule, []