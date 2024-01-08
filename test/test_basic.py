import concurrent.futures
import time

import caws
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.predictors.predictor import Predictor

from test.util_funcs import * # Imports test functions

def test_one(endpoint_id):
    
    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitoring_avail=True)

    endpoints = [endpoint]
    strategy = FCFS_RoundRobin(endpoints, Predictor(endpoints, "sqlite:///caws_tasks.db"))
    with caws.CawsExecutor(endpoints, strategy) as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3

def test_multi(endpoint_id):

    endpoints = [
        caws.Endpoint(
            "caws-dev",
            endpoint_id,
            monitoring_avail=True,
        ),
    ]
    strategy = FCFS_RoundRobin(endpoints, Predictor(endpoints, "sqlite:///caws_tasks.db"))

    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_tasks.db") as executor:
        # Submit all tasks
        futures = []
        for i in range(100):
            futures.append(executor.submit(sleep, 5))

        concurrent.futures.wait(futures)
