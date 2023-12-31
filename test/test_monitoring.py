import pytest
import logging
import os

logger = logging.getLogger(__name__)

from globus_compute_sdk import Executor

import caws
from caws.predictors.predictor import Predictor
from caws.strategy.round_robin import FCFS_RoundRobin
from test.util_funcs import sleep

def test_collection_funcx(endpoint_id):
    # Check to make sure we can collect monitoring data
    # Probably want a test like this inside funcx as well
    # but I think it's good to include here too
    with Executor(endpoint_id=endpoint_id, 
                  monitoring=True) as gce:

        future = gce.submit(sleep, 5)
        future.result()

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
    endpoints = [endpoint,]
    strategy = FCFS_RoundRobin(endpoints, Predictor(endpoints, "sqlite:///caws_tasks.db"))
    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_monitoring.db") as executor:
        fut = executor.submit(sleep, 5)
        fut.result()

    task_df, resource_df, energy_df = endpoint.collect_monitoring_info()

    assert len(task_df) >= 0
    assert len(resource_df) >= 0
    assert len(energy_df) >= 0
