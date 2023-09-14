import os
import math

import caws
from caws import CawsTaskInfo
from caws.predictors.predictor import Predictor, Prediction

def test_predictor():
    caws_db_url = os.environ["ENDPOINT_MONITOR_DEFAULT"]

    endpoint = caws.Endpoint(
        "theta",
        compute_id="14d17201-7380-4af8-b4e0-192cb9805274",
        transfer_id="9032dd3a-e841-4687-a163-2720da731b5b"
    )

    predictor = Predictor([endpoint,], caws_db_url)

    task = CawsTaskInfo(None, (), {}, None, "__main__.graph_bfs")
    result = predictor.predict(endpoint, task)

    assert isinstance(result, Prediction)
    assert not math.isnan(result.runtime)
    assert not math.isnan(result.energy)

if __name__ == "__main__":
    test_predictor()