import os
import time
from datetime import datetime

import caws
from caws.database import CawsDatabase, CawsDatabaseManager
from caws.task import CawsTaskInfo
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor
from util_funcs import add

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

def test_db_manager():
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
    dbm.send_monitoring_message(task_info)
    
    dbm.shutdown()
    assert not dbm.started

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c == 1

def test_integration(endpoint_id):
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring.db"):
        os.remove("caws_monitoring.db")

    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitoring_avail=False)

    endpoints = [endpoint]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_monitoring.db") as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c >= 0

    

    

    
