import os
import math

import caws
from caws import CawsTaskInfo
from caws.predictors.predictor import Predictor, Prediction
from caws.predictors.transfer_predictors import TransferPredictor
from caws.database import CawsDatabaseManager
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.path import CawsPath
from caws.transfer import TransferManager
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

def test_predictor_transfer():
    dbm = CawsDatabaseManager(caws_database_url)
    dbm.start()

    source_name = "caws-dev_ep1"
    source_compute_id = "6754af96-7afa-4c81-b7ef-cf54587f02fa"
    source_transfer_id = "12906d72-48e0-11ee-8135-15041d20ea55"

    dest_name = "caws-dev_ep2"
    dest_compute_id = "ef86d1de-d421-4423-9c6f-09acbfabf5b6"
    dest_transfer_id = "2fde89c0-6fb4-11eb-8c47-0eb1aa8d4337"

    source = caws.Endpoint(source_name,
                           source_compute_id,
                           endpoint_path="/home/alokvk2/research",
                           transfer_id=source_transfer_id,
                           monitoring_avail=True)

    destination = caws.Endpoint(dest_name,
                                dest_compute_id,
                                endpoint_path="/scratch/midway3/alokvk2/",
                                transfer_id=dest_transfer_id,
                                monitoring_avail=True)
    msgs = []
    for endpoint in [source, destination]:
        msg = {
            "endpoint_id": endpoint.compute_endpoint_id,
            "transfer_endpoint_id": endpoint.transfer_endpoint_id,
            "tasks_run": 0,
            "energy_consumed": 0
        }
        msgs.append(msg)
    dbm.update_endpoints(msgs)
    predictor = Predictor([source, destination], caws_database_url)
    predictor.start()

    def success_callback():
        assert True

    def failure_callback(transfer_record):
        assert False, f"Transfer failed: {transfer_record.error}"

    transfer_manager = TransferManager(caws_db=dbm)
    transfer_manager.start()

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "hello_world.txt")
    caws_path = CawsPath(source, file_path)
    caws_path_isolated = CawsPath(source, file_path, isolate=True)
    transfer_record = transfer_manager.transfer([caws_path,],
                              destination,
                              "test_transfer",
                              success_callback,
                              failure_callback)
    transfer_manager.submit_pending_transfers()
    transfer_manager.shutdown()
    dbm.shutdown()

    predictor.update(source)
    result = predictor.predict_transfer(source, destination, caws_path.size, 1)
    assert not math.isnan(result.runtime)
    assert not math.isnan(result.energy)



if __name__ == "__main__":
    test_predictor_transfer()
