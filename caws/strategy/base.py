from typing import List, Tuple
from collections import defaultdict
import copy
from abc import ABC

from caws import Endpoint, CawsTaskInfo
from caws.predictors.predictor import Predictor

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

    def __init__(self, endpoints: List[Endpoint], predictor: Predictor):
        
        if len(endpoints) == 0:
            raise ValueError("List of endpoints cannot be empty")

        self.endpoints = {e.name: e for e in endpoints}
        self.predictor = predictor

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

    def calculate_transfer(self, schedule):
        aggregate_transfer_size = defaultdict(int)
        aggregate_transfer_files = defaultdict(int)
        for task, dst_endpoint in schedule:
            for src_endpoint_id in task.transfer_size.keys():
                aggregate_transfer_size[(src_endpoint_id, dst_endpoint.transfer_endpoint_id)] += task.transfer_size[src_endpoint_id]
                aggregate_transfer_files[(src_endpoint_id, dst_endpoint.transfer_endpoint_id)] += task.transfer_files[src_endpoint_id]

        total_runtime = 0
        total_energy = 0
        for (pair, size), files in zip(aggregate_transfer_size.items(), aggregate_transfer_files.values()):
            pred = self.predictor.predict_transfer(*pair, size, files)
            total_runtime = max(total_runtime, pred.runtime)
            total_energy += pred.energy
            
        return total_runtime, total_energy

    def __str__(self):
        return type(self).__name_

class Schedule:
    endpoint_to_task: dict[str, List]
    name_to_endpoint: dict[str, Endpoint]

    def __init__(self):
        self.endpoint_to_task = defaultdict(list)
        self.name_to_endpoint = dict()

    def __iter__(self):
        schedule = []
        for (e, tasks) in self.endpoint_to_task.items():
            for task in tasks:
                schedule.append((task, self.name_to_endpoint[e]))
        return iter(schedule)
    
    def copy(self):
        s = Schedule()
        s.name_to_endpoint = self.name_to_endpoint
        # Copy dict and lists, but not tasks
        for endpoint_name, tasks in self.endpoint_to_task.items():
            s.endpoint_to_task[endpoint_name] = tasks.copy()
        return s
    
    def add_task(self, endpoint, task):
        s = self.copy()
        s.endpoint_to_task[endpoint.name].append(task)
        if endpoint.name not in self.name_to_endpoint:
            s.name_to_endpoint[endpoint.name] = endpoint
        return s
