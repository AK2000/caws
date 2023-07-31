import pytest

from globus_compute_sdk import Executor

def add(a: int, b: int):
    return a + b

def test_funcx():
    # Check to make sure regular functionality of funcx and the endpoint
    # If this fails, nothing else will work
    with Executor(endpoint_id="27c10959-3ae1-4ce9-a20d-466e7bba293c") as gce:
        future = gce.submit(add, 10, 5)
        assert future.result() == 15

if __name__ == "__main__":
    test_funcx()