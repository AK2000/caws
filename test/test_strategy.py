from collections import defaultdict
from dataclasses import dataclass, field
import random
import uuid

from caws.strategy.mhra import MHRA, MockEndpoint
from caws.strategy.cluster_mhra import ClusterMHRA
from caws.endpoint import EndpointState
from caws.predictors.predictor import Prediction
from caws.task import CawsTaskInfo

random.seed(0)

class MockPredictor:
    def __init__(self, endpoints):
        self.endpoint_static_power = defaultdict(lambda: random.uniform(100, 200))
        self.endpoint_task_runtime = defaultdict(lambda: defaultdict(lambda: random.uniform(0, 30)))
        self.task_avg_power = defaultdict(lambda: random.uniform(0,10))

    def predict_execution(self, endpoint, task):
        runtime = self.endpoint_task_runtime[endpoint.name][task.function_name]
        energy = self.task_avg_power[task.function_name] * runtime
        return Prediction(runtime, energy)

    def predict_transfer(self, src_endpoint, dst_endpoint, size, files):
        return Prediction(3 * files, size * .05)

    def predict_static_power(self, endpoint):
        return self.endpoint_static_power[endpoint.name]

    def predict_cold_start(self, endpoint):
        return 10


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
    shutdown_time: int = 5
    scheduled_tasks: set = field(default_factory=set)

def test_strategy_mock_endpoint():
    endpoint = Endpoint("Endpoint1", 2, 0, 0, 1, EndpointState.COLD)
    predictor = MockPredictor([endpoint])
    task_runtime, task_energy = predictor.predict(endpoint, "task1")
    mock_endpoint = MockEndpoint(endpoint, predictor.static_power(endpoint))
    
    for i in range(5):
        mock_endpoint.schedule(task_runtime, task_energy)

    correct_energy = (task_energy * 5) + (predictor.static_power(endpoint) * (mock_endpoint.runtime + 30))
    assert mock_endpoint.runtime == (task_runtime * 3)
    assert mock_endpoint.energy() - correct_energy < 0.1

def test_strategy_mock_endpoint_2():
    endpoints = [
        Endpoint("Endpoint1", 64, 0, 0, 1, EndpointState.COLD),
        Endpoint("Endpoint2", 32, 0.5, 0, 5, EndpointState.COLD),
        Endpoint("Endpoint3", 16, 0, 1, 1, EndpointState.WARM, 16)
    ]
    tasks = []
    for task_name in ["task1", "task2", "task3", "task4", "task5"]:
        for _ in tasks[::15]:
            task_id = uuid.uuid4()
            tasks.append(CawsTaskInfo(lambda : 0, [], {}, task_id, task_name))

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
    tasks = []
    for task_name in ["task1", "task2", "task3", "task4", "task5"]:
        for _ in range(15):
            task_id = uuid.uuid4()
            tasks.append(CawsTaskInfo(lambda : 0, [], {}, task_id, task_name))

    predictor = MockPredictor(endpoints)
    print("Task Runtime and Energy Consumptions")
    for t in tasks[::15]:
        print(f"Task {t.function_name}")
        print("\t".join([str(predictor.predict_execution(e, t)[0]) for e in endpoints]))
        print("\t".join([str(predictor.predict_execution(e, t)[1]) for e in endpoints]))
    
    print("Endpoint Static Power Consumption:")
    print("\t".join([str(predictor.predict_static_power(e)) for e in endpoints]))

    strategy = MHRA(endpoints, predictor, alpha=1.0)
    schedule, _ = strategy.schedule(tasks)

    endpoint_count = defaultdict(int)
    for task, endpoint in schedule:
        endpoint_count[endpoint.name] += 1

    print("Generated Schedule: ")
    for e, v in endpoint_count.items():
        print(f"Number of tasks on endpoint {e}: {v}")

    assert True

def test_strategy_cluster_mhra():
    endpoints = [
        Endpoint("Endpoint1", 64, 0, 0, 1, EndpointState.COLD),
        Endpoint("Endpoint2", 32, 0.5, 0, 5, EndpointState.COLD),
        Endpoint("Endpoint3", 16, 0, 0, 1, EndpointState.WARM, 16)
    ]
    endpoints = endpoints[:1]
    tasks = []
    for task_name in ["task1", "task2", "task3", "task4", "task5"]:
        for _ in range(15):
            task_id = uuid.uuid4()
            tasks.append(CawsTaskInfo(lambda : 0, [], {}, task_id, task_name))

    predictor = MockPredictor(endpoints)
    print("Task Runtime and Energy Consumptions")
    for t in tasks[::15]:
        print(f"Task {t.function_name}")
        print("\t".join([str(predictor.predict_execution(e, t)[0]) for e in endpoints]))
        print("\t".join([str(predictor.predict_execution(e, t)[1]) for e in endpoints]))
    
    print("Endpoint Static Power Consumption:")
    print("\t".join([str(predictor.predict_static_power(e)) for e in endpoints]))

    strategy = ClusterMHRA(endpoints, predictor, alpha=0.5)

    schedule, _ = strategy.schedule(tasks)

    endpoint_count = defaultdict(int)
    for task, endpoint in schedule:
        endpoint_count[endpoint.name] += 1

    print("Generated Schedule: ")
    for e, v in endpoint_count.items():
        print(f"Number of tasks on endpoint {e}: {v}")

    assert True

if __name__ == "__main__":
    test_strategy_cluster_mhra()