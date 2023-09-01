import pytest

import caws

def test_carbon(endpoint_id):
    endpoint = caws.Endpoint("caws-dev", 
                             endpoint_id,
                             monitor_carbon=True)

    carbon_history = endpoint.collect_carbon_intensity()
    assert carbon_history is not None and carbon_history["history"]