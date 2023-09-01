import pytest
import logging
import os

logger = logging.getLogger(__name__)

from globus_compute_sdk import Executor

import caws
from utils import mainify

def add(a: int, b: int):
    import time
    time.sleep(3) # wait overly long to ensure a resource message has been captured
    return a + b
add = mainify(add)

def test_collection_funcx(endpoint_id):
    # Check to make sure we can collect monitoring data
    # Probably want a test like this inside funcx as well
    # but I think it's good to include here too
    with Executor(endpoint_id=endpoint_id, 
                  monitoring=True, 
                  monitor_resources=True,
                  resource_monitoring_interval=1) as gce:

        future = gce.submit(add, 10, 5)
        assert future.result() == 15

    import sqlalchemy
    from sqlalchemy import text

    logger.info("checking database content")
    engine = sqlalchemy.create_engine(os.environ["ENDPOINT_MONITOR_DEFAULT"])
    with engine.begin() as connection:

        result = connection.execute(text("SELECT COUNT(*) FROM workflow"))
        (c, ) = result.first()
        assert c >= 1

        result = connection.execute(text("SELECT COUNT(*) FROM task"))
        (c, ) = result.first()
        assert c >= 1

        result = connection.execute(text("SELECT COUNT(*) FROM try"))
        (c, ) = result.first()
        assert c >= 1

        # Two entries: one showing manager active, one inactive
        result = connection.execute(text("SELECT COUNT(*) FROM node"))
        (c, ) = result.first()
        assert c >= 2

        # There should be one block polling status
        # local provider has a status_polling_interval of 5s
        result = connection.execute(text("SELECT COUNT(*) FROM block"))
        (c, ) = result.first()
        assert c >= 2

        result = connection.execute(text("SELECT COUNT(*) FROM resource"))
        (c, ) = result.first()
        assert c >= 1

    logger.info("all done")

def test_collection_endpoint(endpoint_id):
    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitoring_avail=True)

    task_df, resource_df, energy_df = endpoint.collect_monitoring_info()

    assert len(task_df) >= 0
    assert len(resource_df) >= 0
    assert len(energy_df) >= 0

if __name__ == "__main__":
    test_collection_endpoint()