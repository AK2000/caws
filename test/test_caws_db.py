import os
import time
from datetime import datetime
import json

import caws
from caws.database import CawsDatabase, CawsDatabaseManager
from caws.task import CawsTaskInfo
from caws.predictors.predictors import Predictor
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.path import CawsPath
from util_funcs import add, transfer_file

def test_database_create():
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring.db"):
        os.remove("caws_monitoring.db")

    CawsDatabase("sqlite:///caws_monitoring.db")

    assert os.path.exists("caws_monitoring.db")

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c == 0

def test_database_manager():
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring.db"):
        os.remove("caws_monitoring.db")
    
    dbm = CawsDatabaseManager("sqlite:///caws_monitoring.db")
    assert not dbm.started
    dbm.start()
    assert dbm.started

    task_info = CawsTaskInfo(add, (1, 2), {}, 0, add.__name__)
    task_info.timing_info["submit"] = datetime.now()
    dbm.send_task_message(task_info)

    transfer_info = {
        "name": "test_transfer_msg",
        "transfer_id": "0",
        "caws_task_id": str(task_info.task_id),
        "src_endpoint_id": "27c10959-3ae1-4ce9-a20d-466e7bba293c",
        "dest_endpoint_id": "ef86d1de-d421-4423-9c6f-09acbfabf5b6",
        "files": json.dumps(["test_file.txt", "test_file_2.txt"]),
        "size": 0,
        "time_submit": datetime.now(),
        "transfer_status": "CREATED"
    }
    dbm.send_transfer_message(transfer_info)
    
    dbm.shutdown()
    assert not dbm.started

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c == 1

        result = connection.execute(text("SELECT COUNT(*) FROM transfer"))
        (c, ) = result.first()
        assert c == 1

def test_database_integration():
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring.db"):
        os.remove("caws_monitoring.db")

    source_path = "test_transfer.txt"
    source_name = "caws-dev_ep1"
    source_compute_id = "27c10959-3ae1-4ce9-a20d-466e7bba293c"
    source_transfer_id = "02ca3fe4-0dee-11ee-bdcc-a3018385fcef"

    dest_name = "caws-dev_ep2"
    dest_compute_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    dest_transfer_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    source = caws.Endpoint(source_name,
                           source_compute_id,
                           endpoint_path="/D/UChicago/src/research",
                           local_path="/mnt/d/UChicago/src/research",
                           transfer_id=source_transfer_id,
                           monitoring_avail=True)

    destination = caws.Endpoint(dest_name,
                                dest_compute_id,
                                endpoint_path="/alokvk2/",
                                local_path="/home/alokvk2/",
                                transfer_id=dest_transfer_id,
                                monitoring_avail=True)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "hello_world.txt")
    caws_path = CawsPath(source, file_path)
    endpoints = [destination, source]
    strategy = FCFS_RoundRobin(endpoints, Predictor(endpoitns, "sqlite:///caws_tasks.db"))
    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_monitoring.db") as executor:
        fut = executor.submit(transfer_file, caws_path)
        assert fut.result()

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c >= 0

        result = connection.execute(text("SELECT COUNT(*) FROM transfer"))
        (c, ) = result.first()
        assert c >= 0

if __name__ == "__main__":
    test_database_integration()