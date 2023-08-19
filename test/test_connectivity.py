import pytest

from globus_compute_sdk import Executor

def add(a: int, b: int):
    return a + b

def test_funcx():
    # Check to make sure regular functionality of funcx and the endpoint
    # If this fails, nothing else will work
    with Executor(endpoint_id="ef86d1de-d421-4423-9c6f-09acbfabf5b6") as gce:
        future = gce.submit(add, 10, 5)
        assert future.result() == 15

if __name__ == "__main__":
    test_funcx()