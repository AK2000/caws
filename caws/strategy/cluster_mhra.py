from collections import defaultdict

from tqdm import tqdm
import numpy as np
from sklearn.cluster import ward_tree

from .base import Strategy, Schedule
from caws.predictors.predictor import Prediction
from caws.endpoint import EndpointState

class MockEndpoint:
    def __init__(self, endpoint, static_power_per_block, cold_start):
        self.endpoint = endpoint

        self.state = endpoint.state
        self.active_tasks = endpoint.active_tasks
        self.active_slots = endpoint.active_slots
        self.slots_per_block = endpoint.slots_per_block
        self.endpoint_state = endpoint.state
        self.min_blocks = endpoint.min_blocks
        self.max_blocks = endpoint.max_blocks
        self.parallelism = endpoint.parallelism
        self.shutdown_time = endpoint.shutdown_time

        self.active_blocks = self.active_slots / self.slots_per_block
        if self.state == EndpointState.WARMING or len(endpoint.scheduled_tasks) > 0:
            self.active_blocks = max(self.active_blocks, 1)
        self.active_blocks = max(self.active_blocks, self.min_blocks)
        
        self.cold_start_time = cold_start

        # List of runtimes by arrival
        self.task_runtimes = []

        self.static_power = static_power_per_block
        self.total_task_energy = 0
        self._runtime = 0
        self.worker_deadlines = [0 for _ in range(self.active_slots)]

    def _calc_runtime(self, task_runtime, worker_deadlines = None):
        if worker_deadlines is None:
            worker_deadlines = self.worker_deadlines

        worker, eft = min(enumerate(worker_deadlines), key=lambda x:x[1])
        new_finish_time = eft + task_runtime
        worker_deadlines[worker] = new_finish_time
        return max(worker_deadlines), worker_deadlines

    def schedule(self, tasks):
        for (task_runtime, task_energy) in tasks:
            self.active_tasks += 1
            if (self.active_slots == 0) or (self.active_slots / self.active_tasks < self.parallelism and self.active_blocks < self.max_blocks):
                self.active_blocks += 1
                self.active_slots += self.slots_per_block

                self.worker_deadlines = [0 for _ in range(self.active_slots)]
                for task in self.task_runtimes:
                    runtime, worker_deadlines = self._calc_runtime(task)
                    self._runtime = runtime
                    self.worker_deadlines = worker_deadlines

            runtime, worker_deadlines = self._calc_runtime(task_runtime)
            self._runtime = runtime
            self.worker_deadlines = worker_deadlines
            self.task_runtimes.append(task_runtime)
            self.total_task_energy += task_energy

    def predict(self, tasks):
        active_slots = self.active_slots
        active_blocks = self.active_blocks
        active_tasks = self.active_tasks
        task_runtimes = self.task_runtimes.copy()
        worker_deadlines = self.worker_deadlines.copy()
        total_task_energy = self.total_task_energy 

        for (task_runtime, task_energy) in tasks:
            if (active_slots == 0) or (active_slots / (active_tasks+1) < self.parallelism and active_blocks < self.max_blocks):
                active_slots = active_slots + self.slots_per_block
                active_blocks = active_blocks + 1
                worker_deadlines = [0 for _ in range(active_slots)]
                for task in task_runtimes:
                    runtime, worker_deadlines = self._calc_runtime(task, worker_deadlines)
                runtime, worker_deadlines = self._calc_runtime(task_runtime, worker_deadlines)
            else:
                runtime, worker_deadlines = self._calc_runtime(task_runtime, worker_deadlines)
            
            total_task_energy += task_energy
            task_runtimes.append(task_runtime)
        
        total_energy = total_task_energy + (runtime * active_blocks * self.static_power)
        total_energy += (active_blocks - self.min_blocks) * self.shutdown_time * self.static_power
        return runtime, total_energy

    def energy(self):
        energy = self.total_task_energy + (self._runtime * self.active_blocks * self.static_power)
        energy += (self.active_blocks - self.min_blocks) * self.shutdown_time * self.static_power
        return energy

    def runtime(self):
        if self.state == EndpointState.COLD and self._runtime > 0:
            return self._runtime + self.cold_start_time
        return self._runtime

def identity(tasks_by_runtime, tasks_by_energy):
    return [t[1] for t in tasks_by_runtime]

def longest_processing_time(tasks_by_runtime, tasks_by_energy):
    tasks_by_runtime = sorted(tasks_by_runtime, key=lambda t: t[0], reverse=True)
    return [t[1] for t in tasks_by_runtime]

def shortest_processing_time(tasks_by_runtime, tasks_by_energy):
    tasks_by_runtime = sorted(tasks_by_runtime, key=lambda t: t[0])
    return [t[1] for t in tasks_by_runtime]

def greatest_energy(tasks_by_runtime, tasks_by_energy):
    tasks_by_energy = sorted(tasks_by_energy, key=lambda t: t[0], reverse=True)
    return [t[1] for t in tasks_by_energy]

def least_energy(tasks_by_runtime, tasks_by_energy):
    tasks_by_energy = sorted(tasks_by_energy, key=lambda t: t[0])
    return [t[1] for t in tasks_by_energy]

class ClusterMHRA(Strategy):

    def __init__(self, 
                 endpoints,
                 predictor,
                 alpha=1.0,
                 heuristics=[identity, longest_processing_time, shortest_processing_time, greatest_energy, least_energy]):
        self.cur_idx = 0
        self.endpoints = endpoints
        self.predictor = predictor
        self.heuristics = heuristics

        self.alpha = alpha

    def objective(self, runtime, energy):
        return ((self.alpha * energy)/ self.energy_normalization) \
                    + (((1 - self.alpha) * runtime)/self.runtime_normalization)

    def preprocess(self, tasks):
        mock_endpoint = MockEndpoint(self.endpoints[0],
                                     self.predictor.predict_static_power(self.endpoints[0]),
                                     self.predictor.predict_cold_start(self.endpoints[0]))
        task_embeddings = np.zeros((len(tasks), len(self.endpoints) * 2))
        print("Preprocessing Tasks")

        task_runtimes = []
        task_energies = []
        task_preds = defaultdict(dict)
        for i, task in tqdm(enumerate(tasks)):
            for j, endpoint in enumerate(self.endpoints):
                task_runtime, task_energy = self.predictor.predict_execution(endpoint, task)
                task_embeddings[i, 2*j] = task_runtime
                task_embeddings[i, (2*j) + 1] = task_energy
                task_preds[task.task_id][endpoint.name] = Prediction(task_runtime, task_energy)

                if j == 0:
                    mock_endpoint.schedule([(task_runtime, task_energy)])
                    task_runtimes.append(task_runtime)
                    task_energies.append(task_energy)
        
        self.runtime_normalization = mock_endpoint.runtime()
        self.energy_normalization = mock_endpoint.energy()

        print(f"On a single endpoint, tasks would take: ")
        print(f"\t{mock_endpoint.runtime()} s")
        print(f"\t{mock_endpoint.energy()} J")

        # TODO: Implement other heuristics
        tasks_by_runtime = zip(task_runtimes, tasks)
        tasks_by_energy = zip(task_energies, tasks)
        return list(tasks_by_runtime), list(tasks_by_energy), task_embeddings, task_preds

    def cluster(self, tasks_by_energy, task_embeddings, threshold=None):
        if threshold is None:
            threshold = 0.25 * max([self.predictor.predict_static_power(e) * e.shutdown_time for e in self.endpoints])
        print(f"Threshold: {threshold}")
        print(len(tasks_by_energy))

        if len(tasks_by_energy) < 3:
            return [tuple(t for _, t in tasks_by_energy)]

        task_embeddings += np.random.uniform(-1, 1, size=task_embeddings.shape)
        children, ncc, n_leaves, parents = ward_tree(task_embeddings)

        clusters = {i: (c, [t]) for i, (c, t) in enumerate(tasks_by_energy)}
        nsamples = len(tasks_by_energy)
        for i in range(children.shape[0]):
            n0 = children[i,0]
            n1 = children[i,1]
            if (n0 in clusters) and (n1 in clusters) and (clusters[n0][0] < threshold and clusters[n1][0] < threshold):
                costs = clusters[n0][0] + clusters[n1][0]
                tasks = clusters[n0][1] + clusters[n1][1]
                clusters[i + nsamples] = (costs, tasks)
                del clusters[n0]
                del clusters[n1]

        return [tuple(tasks) for _, tasks in clusters.values()]
    
    def schedule(self, tasks):
        tasks_by_runtime, tasks_by_energy, task_embeddings, task_preds = self.preprocess(tasks)
        tasks = longest_processing_time(tasks_by_runtime, tasks_by_energy)
        clusters = self.cluster(tasks_by_energy, task_embeddings)
        print(len(clusters))

        cluster_preds = defaultdict(dict)
        clusters_by_runtime = []
        clusters_by_energy = []
        for cluster_id, cluster in enumerate(clusters):
            print([t.function_name for t in cluster])
            for i, endpoint in enumerate(self.endpoints):
                cluster_preds[cluster_id][endpoint.name] = [task_preds[t.task_id][endpoint.name] for t in cluster]
                
                if i == 0:
                    clusters_by_runtime.append(
                        (sum(p.runtime for p in cluster_preds[cluster_id][endpoint.name]), 
                        cluster_id))
                    clusters_by_energy.append(
                        (sum(p.energy for p in cluster_preds[cluster_id][endpoint.name]),
                        cluster_id))

        best_cost = float("inf")
        for heuristic in self.heuristics:
            cluster_ids = heuristic(clusters_by_runtime, clusters_by_energy)

            mock_endpoints = [MockEndpoint(e, 
                                           self.predictor.predict_static_power(e),
                                           self.predictor.predict_cold_start(e)) for e in self.endpoints]
            cur_schedule = Schedule()
            cur_cost = 0

            for cluster_id in cluster_ids:
                cluster = clusters[cluster_id]

                mock_endpoint = mock_endpoints[0]
                cluster_costs = cluster_preds[cluster_id][mock_endpoint.endpoint.name]
                endpoint_runtime, endpoint_energy = mock_endpoint.predict(cluster_costs)
                makespan_runtime = max([e.runtime() for e in mock_endpoints[1:]] + [endpoint_runtime])
                makespan_energy = sum([e.energy() for e in mock_endpoints[1:]]+[endpoint_energy])

                best_endpoint = mock_endpoint
                new_schedule = cur_schedule.add_task(mock_endpoint.endpoint, cluster[0])
                for task in cluster[1:]:
                    new_schedule = new_schedule.add_task(mock_endpoint.endpoint, task)
                transfer_runtime, transfer_energy = self.calculate_transfer(new_schedule)
                makespan_energy += transfer_energy

                new_cost = self.objective(makespan_runtime, makespan_energy)
                best_costs = cluster_costs

                for i, mock_endpoint in enumerate(mock_endpoints[1:], start=1):
                    cluster_costs = cluster_preds[cluster_id][mock_endpoint.endpoint.name]
                    endpoint_runtime, endpoint_energy = mock_endpoint.predict(cluster_costs)
                    makespan_runtime = max(*[e.runtime() for e in mock_endpoints[:i]],
                                        *[e.runtime() for e in mock_endpoints[i+1:]],
                                        endpoint_runtime)
                    makespan_energy = sum([e.energy() for e in mock_endpoints[:i]] \
                                        + [e.energy() for e in mock_endpoints[i+1:]] \
                                        + [endpoint_energy])

                    temp_schedule = cur_schedule.add_task(mock_endpoint.endpoint, cluster[0])
                    for task in cluster[1:]:
                        temp_schedule = temp_schedule.add_task(mock_endpoint.endpoint, task)
                    transfer_runtime, transfer_energy = self.calculate_transfer(temp_schedule)
                    makespan_energy += transfer_energy

                    cost = self.objective(makespan_runtime, makespan_energy)
                    if cost < new_cost:
                        best_endpoint = mock_endpoint
                        new_schedule = temp_schedule
                        new_cost = cost
                        best_costs = cluster_costs

                cur_schedule = new_schedule
                cur_cost = new_cost
                best_endpoint.schedule(cluster_costs)

            print(f"With heuristic {heuristic.__name__}, tasks would take: ")
            makespan_runtime = max([e.runtime() for e in mock_endpoints])
            makespan_energy = sum([e.energy() for e in mock_endpoints])
            transfer_runtime, transfer_energy = self.calculate_transfer(cur_schedule)
            makespan_energy += transfer_energy
            cost = self.objective(makespan_runtime, makespan_energy)
            print(f"\t{makespan_runtime} s")
            print(f"\t{makespan_energy/1e9} kJ")
            print(f"\tCost: {cost}")
            print(f"\tTransfer Energy: {transfer_energy/1e9} kJ")

            if cur_cost < best_cost:
                best_schedule = cur_schedule
                best_cost = cur_cost
                best_runtime = makespan_runtime
                best_energy = makespan_energy

        print(f"After scheduling, tasks would take: ")
        print(f"\t{best_runtime} s")
        print(f"\t{best_energy/1e9} kJ")

        endpoint_count = defaultdict(int)
        for task, endpoint in best_schedule:
            endpoint_count[endpoint.name] += 1

        print("Generated Schedule: ")
        for e, v in endpoint_count.items():
            print(f"Number of tasks on endpoint {e}: {v}")

        return best_schedule, []
