from .base import Strategy, Schedule

class MockEndpoint:
    def __init__(self, endpoint, static_power_per_block):
        self.endpoint = endpoint

        self.active_tasks = endpoint.active_tasks
        self.active_slots = endpoint.active_slots
        self.slots_per_block = endpoint.slots_per_block
        self.endpoint_state = endpoint.state
        self.min_blocks = endpoint.min_blocks
        self.max_blocks = endpoint.max_blocks
        self.parallelism = endpoint.parallelism

        self.active_blocks = self.active_slots / self.slots_per_block

        # List of runtimes by arrival
        self.task_runtimes = []

        self.static_power = static_power_per_block
        self.total_task_energy = 0
        self.runtime = 0
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
                self.runtime = runtime
                self.worker_deadlines = worker_deadlines

        runtime, worker_deadlines = self._calc_runtime(task_runtime)
        self.runtime = runtime
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

        return runtime, total_energy

    def energy(self):
        energy = self.total_task_energy + (self.runtime * self.active_blocks * self.static_power)
        return energy

class MHRA(Strategy):

    def __init__(self, endpoints, predictor, alpha=1.0):
        self.cur_idx = 0
        self.endpoints = endpoints
        self.predictor = predictor

        self.alpha = alpha

    def objective(self, runtime, energy):
        return ((self.alpha * energy)/ self.energy_normalization) \
                    + (((1 - self.alpha) * runtime)/self.runtime_normalization)

    def preprocess(self, tasks):
        mock_endpoint = MockEndpoint(self.endpoints[0], self.predictor.static_power(self.endpoints[0]))
        print("Preprocessing Tasks")

        task_runtimes = []
        for task in tasks:
            task_runtime, task_energy = self.predictor.predict(mock_endpoint.endpoint, task)
            mock_endpoint.schedule(task_runtime, task_energy)

            task_runtimes.append(task_runtime)
        
        self.runtime_normalization = mock_endpoint.runtime
        self.energy_normalization = mock_endpoint.energy()

        print(f"On a single endpoint, tasks would take: ")
        print(f"\t{mock_endpoint.runtime} s")
        print(f"\t{mock_endpoint.energy()} J")

        # TODO: Implement other heuristics
        tasks_by_runtime = sorted(zip(task_runtimes, tasks), reverse=True)
        return [t[1] for t in tasks_by_runtime]

    def schedule(self, tasks):
        tasks = self.preprocess(tasks)
        mock_endpoints = [MockEndpoint(e, self.predictor.static_power(e)) for e in self.endpoints]
        best_schedule = Schedule()
        best_cost = 0
        best_runtime = 0
        best_energy = 0

        for task in tasks:
            mock_endpoint = mock_endpoints[0]
            task_runtime, task_energy = self.predictor.predict(mock_endpoint.endpoint, task)
            endpoint_runtime, endpoint_energy = mock_endpoint.predict(task_runtime, task_energy)
            makespan_runtime = max(*[e.runtime for e in mock_endpoints[1:]], endpoint_runtime)
            makespan_energy = sum([e.energy() for e in mock_endpoints[1:]]+[endpoint_energy])

            best_endpoint = mock_endpoint
            new_schedule = best_schedule.add_task(mock_endpoint.endpoint, task)
            new_cost = self.objective(makespan_runtime, makespan_energy)
            best_task_runtime = task_runtime
            best_task_energy = task_energy
            best_runtime = makespan_runtime
            best_energy = makespan_energy

            for i, mock_endpoint in enumerate(mock_endpoints[1:], start=1):
                task_runtime, task_energy = self.predictor.predict(mock_endpoint.endpoint, task)
                endpoint_runtime, endpoint_energy = mock_endpoint.predict(task_runtime, task_energy)
                makespan_runtime = max(*[e.runtime for e in mock_endpoints[:i]],
                                       *[e.runtime for e in mock_endpoints[i+1:]],
                                       endpoint_runtime)
                makespan_energy = sum([e.energy() for e in mock_endpoints[:i]] \
                                      + [e.energy() for e in mock_endpoints[i+1:]] \
                                      + [endpoint_energy])

                cost = self.objective(makespan_runtime, makespan_energy)
                if cost < new_cost:
                    best_endpoint = mock_endpoint
                    new_schedule = best_schedule.add_task(mock_endpoint.endpoint, task)
                    new_cost = cost
                    best_task_runtime = task_runtime
                    best_task_energy = task_energy
                    best_runtime = makespan_runtime
                    best_energy = makespan_energy

            best_schedule = new_schedule
            best_cost = new_cost
            best_endpoint.schedule(best_task_runtime, best_task_energy)

        print(f"After scheduling, tasks would take: ")
        print(f"\t{best_runtime} s")
        print(f"\t{best_energy} J")

        return best_schedule