from abc import ABC
import time
from collections import defaultdict
from typing import List, Tuple

from caws.endpoint import Endpoint
from caws.predictors.transfer_predictors import TransferPredictor
from caws.task import CawsTaskInfo


FUNCX_LATENCY = 0.1  # Estimated overhead of executing task

class Strategy(ABC):
    """ Strategy interface to provide scheduling decisions

    Parameters
    ----------

    endpoints : List[Endpoints]
        The list of endpoints that can be scheduled. Provides an interface for
        esitmating runtime, energy, carbon, and queue delay. See endpoint class for more
        details. Must be a superset of all endpoints that are scheduled on.

    trasfer_predictor: TransferPredictor
        Model used to estimate transfers between sites. Will be 
        periodically updated by the executor
    """

    def __init__(self, endpoints: List[Endpoint], transfer_predictor: TransferPredictor):
        
        if len(endpoints) == 0:
            raise ValueError("List of endpoints cannot be empty")
        assert(callable(transfer_predictor))

        self.endpoints = {e.name: e for e in endpoints}
        self.transfer_predictor = transfer_predictor

    def schedule(self, tasks: CawsTaskInfo) -> (Tuple, List[CawsTaskInfo]):
        """ Map tasks to endpoints, with the assumption that the endpoint will
        run the task ASAP once it is scheduled.

        #TODO: Figure out how to include a per-task resouce mapping/allowable endpoints
        
        Parameters
        ----------
        tasks : List[CawsTaskInfo]

        
        Returns
        -------
        List[Tuple[CawsTaskInfo, Endpoint]] List[CawsTaskInfo]
            The first object represents a mapping from tasks to endpoints. This is the schedule
            output by the strategy. 

            The second list represents any tasks which the strategy chose to defer to some later time
            (i.e. because it predicts that it will be more "efficient" later)

            #TODO: Change this to a defined object/typed dictionary
        """

        raise NotImplementedError

    def add_endpoint(self, endpoint):
        self.endpoints[endpoint.name] = endpoint

    def remove_endpoint(self, endpoint_name):
        if endpoint_name in self.endpoints:
            del self.endpoints[endpoint_name]
    
    def update(self, task_info):
        raise NotImplementedError

    def __str__(self):
        return type(self).__name__

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