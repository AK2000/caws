import concurrent.futures
import time

import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor

from utils import mainify

def add(a: int, b: int):
    return a + b
add = mainify(add)

def sleep(sec: float):
    import time
    time.sleep(sec)
    return True
sleep = mainify(sleep)

def compute_iters(a: int):
    result = 0
    for i in range(a):
        result += i

    return result
compute_iters = mainify(compute_iters)

def gemm(dim: int):
    import numpy as np

    A = np.random.rand(dim, dim)
    B = np.random.rand(dim, dim)

    return A @ B
gemm = mainify(gemm)

def test_one(endpoint_id):
    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitoring_avail=True)

    endpoints = [endpoint]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
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
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))

    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_tasks.db") as executor:
        # Submit all tasks
        futures = []
        for i in range(100):
            futures.append(executor.submit(compute_iters, 100000))
        for i in range(100):
            futures.append(executor.submit(sleep, 5))
        for i in range(100):
            futures.append(executor.submit(gemm, 256))

        concurrent.futures.wait(futures)