import pytest

from globus_compute_sdk import Executor
from util_funcs import add

def test_funcx(endpoint_id):
    # Check to make sure regular functionality of funcx and the endpoint
    # If this fails, nothing else will work
    with Executor(endpoint_id=endpoint_id) as gce:
        future = gce.submit(add, 10, 5)
        assert future.result() == 15