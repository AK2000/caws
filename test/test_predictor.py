import os
import math

import caws
from caws import CawsTaskInfo
from caws.predictors.predictor import Predictor, Prediction
from caws.database import CawsDatabase, CawsDatabaseManager
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.path import CawsPath
from util_funcs import add, transfer_file

caws_database_url = "test_data/monitoring.db"

def test_predictor_empty():
    caws_db = CawsDatabase(caws_database_url)
    endpoint = caws.Endpoint(
        "desktop",
        compute_id="6754af96-7afa-4c81-b7ef-cf54587f02fa",
        transfer_id="12906d72-48e0-11ee-8135-15041d20ea55"
    )

    msg = {
        "endpoint_id": endpoint.endpoint_id,
        "transfer_endpoint_id": endpoint.transfer_endpoint_id,
        "tasks_run": 0,
        "energy_consumed": 0
    }
    caws_db.update_endpoints([msg])

    predictor = Predictor([endpoint,], caws_database_url)
    task = CawsTaskInfo(None, (), {}, None, "__main__.graph_bfs")

    assert predictor.predict_execution(endpoint, task) is None
    assert predictor.predict_static_power(endpoint) is None
    assert predictor.predict_cold_start(endpoint) == 0


def test_predictor_update():
    pass

def test_predictor_execution():
    pass


def test_predictor_transfer():
    pass

if __name__ == "__main__":
    test_predictor()