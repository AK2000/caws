import os
from datetime import datetime

import caws
from caws.database import CawsDatabaseManager
from caws.task import caws_task, CawsTask, CawsTaskInfo
from caws.features import CawsFeatureType, ArgFeature
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.predictors.predictor import Predictor

import test.util_funcs

def test_feature_basic():
    gemm = caws_task(test.util_funcs.gemm, [ArgFeature(0)]) 
    assert isinstance(gemm, CawsTask)

    features = gemm.extract_features(64)
    assert len(features) == 1
    assert features[0] == (64, CawsFeatureType.CONTINUOUS)

    gemm = caws_task(test.util_funcs.gemm, [ArgFeature(arg_name="dim")])
    features = gemm.extract_features(64)
    assert features[0] == (64, CawsFeatureType.CONTINUOUS)

def test_feature_send():
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring.db"):
        os.remove("caws_monitoring.db")
    
    dbm = CawsDatabaseManager("sqlite:///caws_monitoring.db")
    assert not dbm.started
    dbm.start()
    assert dbm.started

    task_info = CawsTaskInfo(test.util_funcs.add, (1, 2), {}, 0, test.util_funcs.add.__name__)
    task_info.timing_info["submit"] = datetime.now()
    dbm.send_task_message(task_info)
    dbm.send_feature_message({
                    "caws_task_id": 0,
                    "feature_id": 0,
                    "feature_type": CawsFeatureType.CONTINUOUS.name,
                    "value": str(-1)
            })
    dbm.send_feature_message({
                    "caws_task_id": 0,
                    "feature_id": 1,
                    "feature_type": CawsFeatureType.CATEGORICAL.name,
                    "value": "red"
            })
    
    dbm.shutdown()
    assert not dbm.started

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c == 1

        result = connection.execute(text("SELECT COUNT(*) FROM features"))
        (c, ) = result.first()
        assert c == 2

def test_feature_integration(endpoint_id):
    import sqlalchemy
    from sqlalchemy import text

    if os.path.exists("caws_monitoring_2.db"):
        os.remove("caws_monitoring_2.db")

    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitoring_avail=True)

    endpoints = [endpoint]
    strategy = FCFS_RoundRobin(endpoints, Predictor(endpoints, "sqlite:///caws_monitoring_2.db"))
    add = caws_task(test.util_funcs.add, [ArgFeature(0)])
    with caws.CawsExecutor(endpoints, strategy, caws_database_url="sqlite:///caws_monitoring_2.db") as executor:
        fut = executor.submit(add, 1, 2)
        assert fut.result() == 3

    engine = sqlalchemy.create_engine("sqlite:///caws_monitoring_2.db")
    with engine.begin() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM caws_task"))
        (c, ) = result.first()
        assert c == 1

        result = connection.execute(text("SELECT COUNT(*) FROM features"))
        (c, ) = result.first()
        assert c == 1

if __name__ == "__main__":
    test_feature_integration("14d17201-7380-4af8-b4e0-192cb9805274")
