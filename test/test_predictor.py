import os
import math

import caws
from caws import CawsTaskInfo
from caws.predictors.predictor import Predictor, Prediction
from caws.predictors.transfer_predictors import TransferPredictor
from caws.database import CawsDatabaseManager
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.path import CawsPath
from util_funcs import add, transfer_file

caws_database_url = os.environ["ENDPOINT_MONITOR_DEFAULT"]

def test_predictor_empty():
    caws_db = CawsDatabaseManager(caws_database_url)
    endpoint = caws.Endpoint(
        "desktop",
        compute_id="6754af96-7afa-4c81-b7ef-cf54587f02fa",
        transfer_id="12906d72-48e0-11ee-8135-15041d20ea55"
    )

    msg = {
        "endpoint_id": endpoint.compute_endpoint_id,
        "transfer_endpoint_id": endpoint.transfer_endpoint_id,
        "tasks_run": 0,
        "energy_consumed": 0
    }
    caws_db.update_endpoints([msg])

    predictor = Predictor([endpoint,], caws_database_url)
    predictor.start()
    
    task = CawsTaskInfo(None, (), {}, None, "__main__.graph_bfs")

    assert predictor.predict_execution(endpoint, task) is None
    assert predictor.predict_static_power(endpoint) is None
    assert predictor.predict_cold_start(endpoint) == 0

def test_predictor_update():
    endpoint = caws.Endpoint(
        "desktop",
        compute_id="6754af96-7afa-4c81-b7ef-cf54587f02fa",
        transfer_id="12906d72-48e0-11ee-8135-15041d20ea55"
    )
    endpoints = [endpoint]

    predictor = Predictor([endpoint,], caws_database_url)
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy, predictor=predictor) as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3
        task_info = fut.task_info
    
    predictor.update(endpoint)
    result = predictor.predict_execution(endpoint, task_info)

    assert not math.isnan(result.runtime)
    assert not math.isnan(result.energy)
    assert not math.isnan(predictor.predict_static_power(endpoint))

def test_predictor_features():
    pass

def test_predictor_transfer():
    pass

if __name__ == "__main__":
    test_predictor_update()
