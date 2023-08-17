import concurrent.futures
import time

import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor

def sleep(sec: float):
    import time
    time.sleep(sec)
    return True

def compute_iters(a: int):
    result = 0
    for i in range(a):
        result += i

    return result

def gemm(dim: int):
    import numpy as np

    A = np.random.rand(dim, dim)
    B = np.random.rand(dim, dim)

    return A @ B

def test_multi():
    compute_endpoint_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    # compute_endpoint_id = "ef86d1de-d421-4423-9c6f-09acbfabf5b6"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoints = [
        caws.Endpoint(
            "midway-funcx-dev",
            compute_endpoint_id,
            transfer_endpoint_id,
            monitoring_avail=True,
        ),
    ]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))

    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_tasks.db") as executor:
        # Give a chance for the manager to start all the workers
        executor.submit(sleep, 10).result()

        # Submit iteration task
        futures = []
        for i in range(100):
            futures.append(executor.submit(compute_iters, 10000000))
        concurrent.futures.wait(futures)

        # Submit sleep tasks
        futures = []
        for i in range(100):
            futures.append(executor.submit(sleep, 10))
        concurrent.futures.wait(futures)

        # Submit gemm tasks
        futures = []
        for i in range(100):
            futures.append(executor.submit(gemm, 512))
        concurrent.futures.wait(futures)

        # TODO: Test some io intensive task

if __name__ == "__main__":
    test_multi()