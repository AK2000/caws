from collections import defaultdict

from .base import Strategy, Schedule
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

        self.active_blocks = max(self.active_slots / self.slots_per_block, self.min_blocks)
        if self.state == EndpointState.WARMING:
            self.active_blocks += 1

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

    def schedule(self, task_runtime, task_energy):
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

    def predict(self, task_runtime, task_energy):
        if (self.active_slots == 0) or (self.active_slots / (self.active_tasks+1) < self.parallelism and self.active_blocks < self.max_blocks):
            active_slots = self.active_slots + self.slots_per_block
            active_blocks = self.active_blocks + 1
            worker_deadlines = [0 for _ in range(active_slots)]
            for task in self.task_runtimes:
                runtime, worker_deadlines = self._calc_runtime(task, worker_deadlines)
            runtime, worker_deadlines = self._calc_runtime(task_runtime, worker_deadlines)
        else:
            active_slots = self.active_slots
            active_blocks = self.active_blocks
            runtime, worker_deadlines = self._calc_runtime(task_runtime, self.worker_deadlines.copy())
        
        total_task_energy = self.total_task_energy + task_energy
        total_energy = total_task_energy + (runtime * active_blocks * self.static_power)
        total_energy += (active_blocks - self.min_blocks) * self.shutdown_time * self.static_power
        return runtime, total_energy

    def energy(self):
        energy = self.total_task_energy + (self._runtime * self.active_blocks * self.static_power)
        energy += (self.active_blocks - self.min_blocks) * self.shutdown_time * self.static_power
        assert energy >= 0, f"Endpoint energy predicted_negative: {self.total_task_energy}, {self._runtime}, {self.static_power}, {self.active_blocks}, {self.min_blocks}"
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


class MHRA(Strategy):

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
        print("Preprocessing Tasks")

        task_runtimes = []
        task_energies = []
        for task in tasks:
            task_runtime, task_energy = self.predictor.predict_execution(mock_endpoint.endpoint, task)
            mock_endpoint.schedule(task_runtime, task_energy)

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
        return list(tasks_by_runtime), list(tasks_by_energy)

    def calculate_transfer(self, schedule):
        aggregate_transfer_size = defaultdict(int)
        aggregate_transfer_files = defaultdict(int)
        for task, dst_endpoint in schedule:
            for src_endpoint_id in task.transfer_size.keys():
                aggregate_transfer_size[(src_endpoint_id, dst_endpoint.transfer_endpoint_id)] += task.transfer_size[src_endpoint_id]
                aggregate_transfer_files[(src_endpoint_id, dst_endpoint.transfer_endpoint_id)] += task.transfer_files[src_endpoint_id]

        total_runtime = 0
        total_energy = 0
        for (pair, size), files in zip(aggregate_transfer_size.items(), aggregate_transfer_files.values()):
            pred = self.predictor.predict_transfer(*pair, size, files)
            total_runtime = max(total_runtime, pred.runtime)
            total_energy += pred.energy

        return total_runtime, total_energy

    def schedule(self, tasks):
        tasks_by_runtime, tasks_by_energy = self.preprocess(tasks)
        best_cost = float("inf")
        for heuristic in self.heuristics:
            tasks = heuristic(tasks_by_runtime, tasks_by_energy)            
            
            mock_endpoints = [MockEndpoint(e, 
                                           self.predictor.predict_static_power(e),
                                           self.predictor.predict_cold_start(e)) for e in self.endpoints]
            cur_schedule = Schedule()
            cur_cost = 0

            for task in tasks:
                mock_endpoint = mock_endpoints[0]

                task_runtime, task_energy = self.predictor.predict_execution(mock_endpoint.endpoint, task)
                endpoint_runtime, endpoint_energy = mock_endpoint.predict(task_runtime, task_energy)
                makespan_runtime = max(*[e.runtime() for e in mock_endpoints[1:]], endpoint_runtime)
                makespan_energy = sum([e.energy() for e in mock_endpoints[1:]]+[endpoint_energy])

                best_endpoint = mock_endpoint
                new_schedule = cur_schedule.add_task(mock_endpoint.endpoint, task)

                transfer_runtime, transfer_energy = self.calculate_transfer(new_schedule)
                makespan_energy += transfer_energy

                new_cost = self.objective(makespan_runtime, makespan_energy)
                best_task_runtime = task_runtime
                best_task_energy = task_energy

                for i, mock_endpoint in enumerate(mock_endpoints[1:], start=1):
                    task_runtime, task_energy = self.predictor.predict_execution(mock_endpoint.endpoint, task)
                    endpoint_runtime, endpoint_energy = mock_endpoint.predict(task_runtime, task_energy)
                    temp_schedule = cur_schedule.add_task(mock_endpoint.endpoint, task)

                    makespan_runtime = max(*[e.runtime() for e in mock_endpoints[:i]],
                                        *[e.runtime() for e in mock_endpoints[i+1:]],
                                        endpoint_runtime)

                    makespan_energy = sum([e.energy() for e in mock_endpoints[:i]] \
                                        + [e.energy() for e in mock_endpoints[i+1:]] \
                                        + [endpoint_energy])
                    transfer_runtime, transfer_energy = self.calculate_transfer(temp_schedule)
                    makespan_energy += transfer_energy

                    #TODO: Deal with transfer runtime!!!!

                    cost = self.objective(makespan_runtime, makespan_energy)
                    if cost < new_cost:
                        best_endpoint = mock_endpoint
                        new_schedule = temp_schedule
                        new_cost = cost
                        best_task_runtime = task_runtime
                        best_task_energy = task_energy

                cur_schedule = new_schedule
                cur_cost = new_cost
                best_endpoint.schedule(best_task_runtime, best_task_energy)

            print(f"With heuristic {heuristic.__name__}, tasks would take: ")
            print(f"\t{max([e.runtime() for e in mock_endpoints])} s")
            print(f"\t{sum([e.energy() for e in mock_endpoints])} J")

            if cur_cost < best_cost:
                best_schedule = cur_schedule
                best_cost = cur_cost
                best_runtime = max([e.runtime() for e in mock_endpoints])
                best_energy = sum([e.energy() for e in mock_endpoints])
                transfer_runtime, transfer_energy = self.calculate_transfer(cur_schedule)
                best_energy += transfer_energy

        print(f"After scheduling, tasks would take: ")
        print(f"\t{best_runtime} s")
        print(f"\t{best_energy} J")

        endpoint_count = defaultdict(int)
        for task, endpoint in best_schedule:
            endpoint_count[endpoint.name] += 1

        print("Generated Schedule: ")
        for e, v in endpoint_count.items():
            print(f"Number of tasks on endpoint {e}: {v}")

        return best_schedule, []
