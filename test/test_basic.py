import concurrent.futures
import time

import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor

def add(a: int, b: int):
    return a + b

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

def test_one():
    compute_endpoint_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoint = caws.Endpoint("wsl-funcx-dev", 
                             compute_endpoint_id,
                             transfer_endpoint_id,
                             monitoring_avail=True)

    endpoints = [endpoint]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy) as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3

def test_multi():
    compute_endpoint_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoints = [
        caws.Endpoint(
            "theta-funcx-dev",
            compute_endpoint_id,
            transfer_endpoint_id,
            monitoring_avail=True,
        ),
    ]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))

    with caws.CawsExecutor(endpoints, strategy) as executor:
        # Warm up endpoint
        fut = executor.submit(sleep, 5)
        fut.result()

        # Capture some idle time
        time.sleep(5)

        # Submit all tasks
        futures = []
        for i in range(100):
            futures.append(executor.submit(compute_iters, 100000))
        for i in range(100):
            futures.append(executor.submit(sleep, 5))
        for i in range(100):
            futures.append(executor.submit(gemm, 256))

        concurrent.futures.wait(futures)

if __name__ == "__main__":
    test_multi()