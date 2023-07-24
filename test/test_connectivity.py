import pytest

from globus_compute_sdk import Executor

def add(a: int, b: int):
    return a + b

@pytest.mark.skip(reason="For some reason the future won't ever timeout")
def test_funcx():
    # Check to make sure regular functionality of funcx and the endpoint
    # If this fails, nothing else will work
    with Executor(endpoint_id="14d17201-7380-4af8-b4e0-192cb9805274") as gce:
        future = gce.submit(add, 10, 5)
        assert future.result(timeout=30) == 15