from collections import defaultdict
from dataclasses import dataclass
import random

from caws.strategy.mhra import MHRA, MockEndpoint
from caws.endpoint import EndpointState

random.seed(0)

class MockPredictor:
    def __init__(self, endpoints):
        self.endpoint_static_power = defaultdict(lambda: random.uniform(100, 200))
        self.endpoint_task_runtime = defaultdict(lambda: defaultdict(lambda: random.uniform(0, 30)))
        self.task_avg_power = defaultdict(lambda: random.uniform(0,10))

    def predict(self, endpoint, task):
        runtime = self.endpoint_task_runtime[endpoint.name][task]
        energy = self.task_avg_power[task] * runtime
        return runtime, energy

    def static_power(self, endpoint):
        return self.endpoint_static_power[endpoint.name]

@dataclass
class Endpoint:
    name: str
    slots_per_block: int
    parallelism: float
    min_blocks: int
    max_blocks: int
    state: EndpointState
    active_slots: int = 0
    active_tasks: int = 0

def test_strategy_mock_endpoint():
    endpoint = Endpoint("Endpoint1", 2, 0, 0, 1, EndpointState.COLD)
    predictor = MockPredictor([endpoint])
    task_runtime, task_energy = predictor.predict(endpoint, "task1")
    mock_endpoint = MockEndpoint(endpoint, predictor.static_power(endpoint))
    
    for i in range(5):
        mock_endpoint.schedule(task_runtime, task_energy)

    correct_energy = (task_energy * 5) + (predictor.static_power(endpoint) * mock_endpoint.runtime)
    assert mock_endpoint.runtime == (task_runtime * 3)
    assert mock_endpoint.energy() == correct_energy

def test_strategy_mock_endpoint_2():
    endpoints = [
        Endpoint("Endpoint1", 64, 0, 0, 1, EndpointState.COLD),
        Endpoint("Endpoint2", 32, 0.5, 0, 5, EndpointState.COLD),
        Endpoint("Endpoint3", 16, 0, 0, 1, EndpointState.WARM, 16)
    ]
    tasks = [t for t in ["task1", "task2", "task3", "task4", "task5"] for _ in range(10)]

    predictor = MockPredictor(endpoints)
    mock_endpoint = MockEndpoint(endpoints[0], predictor.static_power(endpoints[0]))
    
    for i, task in enumerate(tasks):
        task_runtime, task_energy = predictor.predict(endpoints[0], task)
        pred_runtime, pred_energy = mock_endpoint.predict(task_runtime, task_energy)
        mock_endpoint.schedule(task_runtime, task_energy)

        assert pred_runtime == mock_endpoint.runtime, f"Predict and schedule don't match at iteration {i}"
        assert pred_energy == mock_endpoint.energy(), f"Predict and schedule don't match at iteration {i}"

def test_strategy_mhra():
    endpoints = [
        Endpoint("Endpoint1", 64, 0, 0, 1, EndpointState.COLD),
        Endpoint("Endpoint2", 32, 0.5, 0, 5, EndpointState.COLD),
        Endpoint("Endpoint3", 16, 0, 0, 1, EndpointState.WARM, 16)
    ]
    tasks = [t for t in ["task1", "task2", "task3", "task4", "task5"] for _ in range(15)]

    predictor = MockPredictor(endpoints)
    strategy = MHRA(endpoints, predictor, alpha=1.0)

    schedule = strategy.schedule(tasks)

    endpoint_count = defaultdict(int)
    for task, endpoint in schedule:
        endpoint_count[endpoint.name] += 1

    print("Generated Schedule: ")
    for e, v in endpoint_count.items():
        print(f"Number of tasks on endpoint {e}: {v}")

    assert True