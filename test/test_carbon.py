import pytest

import caws

def test_carbon():
    compute_endpoint_id = "27c10959-3ae1-4ce9-a20d-466e7bba293c"
    transfer_endpoint_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    endpoint = caws.Endpoint("wsl-funcx-dev", 
                             compute_endpoint_id,
                             transfer_endpoint_id,
                             monitor_carbon=True)

    carbon_history = endpoint.collect_carbon_intensity()
    assert carbon_history is not None and carbon_history["history"]

if __name__ == "__main__":
    test_carbon()