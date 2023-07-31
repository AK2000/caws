import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor

@caws.caws_task
def add(a: int, b: int):
    return a + b

def test_one():
    compute_endpoint_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoints = [caws.Endpoint("theta-funcx-dev", compute_endpoint_id, transfer_endpoint_id, monitoring_avail=False),]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy) as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3

def test_multi():
    compute_endpoint_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoints = [caws.Endpoint("theta-funcx-dev", compute_endpoint_id, transfer_endpoint_id, monitoring_avail=False),]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy) as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3
