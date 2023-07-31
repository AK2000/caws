import sys
import time
import json
import uuid
import logging
import requests
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from threading import Thread, Lock, Event
from collections import defaultdict

from caws.utils import client
from caws.transfer import TransferManager
from caws.strategy import Strategy
from caws.task import CawsTaskInfo, TaskStatus, CawsTask, CawsFuture
from caws.endpoint import Endpoint, EndpointState
from caws.predictors.transfer_predictors import TransferPredictor

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    "[SCHEDULER] %(message)s", 'yellow'))
logger.addHandler(ch)

class CawsExecutor(object):
    """ Executor that schedulies tasks based on a carbon aware schedule.
    Currently designed to use seperately from Parsl, but eventually meant
    to be somewhat integrated into Parsl/dataflow kernel.
    """

    endpoints: list[Endpoint]
    strategy: Strategy
    _task_watchdog_sleep: float = 0.2
    _endpoint_watchdog_sleep: float = 5
    _task_scheduling_sleep: float = 0.5

    ready_tasks_lock = Lock()
    ready_tasks: list[CawsTaskInfo] = []
    awaiting_transfer: Queue[CawsTaskInfo] = Queue()

    def __init__(self, 
                 endpoints: list[Endpoint],
                 strategy: Strategy,
                 task_watchdog_sleep: float = 0.2,
                 endpoint_watchdog_sleep: float = 5,
                 task_scheduling_sleep: float = 5):
        
        self.endpoints = endpoints
        self.strategy = strategy

        self._task_watchdog_sleep = task_watchdog_sleep
        self._endpoint_watchdog_sleep = endpoint_watchdog_sleep
        self._task_scheduling_sleep = task_scheduling_sleep

        self._transfer_manger = TransferManager(endpoints)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.shutdown()
        return False

    def start(self):
        print("Executor starting")

        self._kill_event = Event()
        self._task_scheduler = Thread(target=self._schedule_tasks_loop, args=(self._kill_event,))
        self._task_scheduler.start()

        self._task_watchdog = Thread(target=self._monitor_tasks, args=(self._kill_event,))
        self._task_watchdog.start()

        self._endpoint_watchdog = Thread(target=self._check_endpoints, args=(self._kill_event,))
        self._endpoint_watchdog.start()

    def shutdown(self):
        self._kill_event.set()
        self._task_scheduler.join()
        self._task_watchdog.join()
        self._endpoint_watchdog.join()

    def submit(self, fn: Callable, *args, **kwargs):

        task_id = str(uuid.uuid4())
        if isinstance(fn, CawsTask):
            function_id = fn.function_id
        else:
            function_id = client.register_function(func)

        task_info = CawsTaskInfo(function_id, args, kwargs, task_id)
        task_info.caws_future = CawsFuture(task_info)
        
        with self.ready_tasks_lock:
            self.ready_tasks.append(task_info)

        return task_info.caws_future

    def _schedule_tasks_loop(self, kill_event):
        logger.info("Starting task-submission thread")

        self.tasks_scheduling = []
        while not kill_event.is_set():
            with self.ready_tasks_lock:
                self.tasks_scheduling.extend(self.ready_tasks)
                self.ready_tasks = []
            
            scheduling_decisions, self.tasks_scheduling = self.strategy.schedule(self.tasks_scheduling)

            for task, endpoint in scheduling_decisions:
                self._start_task(task, endpoint)

            if len(scheduling_decisions) == 0:
                time.sleep(self._task_scheduling_sleep)

    def _start_task(self, task, endpoint):

        task.task_status = TaskStatus.SCHEDULED

        files = task.task_kwargs.get("_globus_files", {}) # Must be dict of {src: path_str}

        # Start Globus transfer of required files, if any
        if len(files) > 0:
            task.transfer_id = self._transfer_manger.transfer(files, endpoint.name,
                                                          task_id)
            endpoint.schedule(task)
            # Put in queue to monitor when transfer has completed
            self.awaiting_transfer.put((task, endpoint))
        else:
            task.task_status = TaskStatus.EXECUTING
            logger.info("Submitting task to endpoint")
            endpoint.submit(task)


    def _monitor_tasks(self, kill_event):
        logger.info('Starting task-watchdog thread')

        scheduled = []

        while not kill_event.is_set():
            n = self.awaiting_transfer.qsize()
            for _ in range(n):
                try:
                    task, endpoint = self._scheduled_tasks.get_nowait()
                except Empty:
                    break

                if task.transfer_id is not None and not self._transfer_manger.is_complete(task.transfer_id):
                    # Task cannot be scheduled, transfers not complete
                    self._scheduled_tasks.put((task, endpoint))
                    continue

                if task.transfer_id is not None:
                    task.timing["transfer_time"] = self._transfer_manger.get_transfer_time(transfer_num)
                    # TODO: Update transfer model

                task.task_status = TaskStatus.EXECUTING
                endpoint.submit(task)

            # Wait before iterating through queue again
            time.sleep(self._task_watchdog_sleep)

    def _check_endpoints(self, kill_event):
        logger.info('Starting endpoint-watchdog thread')

        while not kill_event.is_set():
            for endpoint in self.endpoints:
                state = endpoint.poll()
                if state == EndpointState.DEAD:
                    self._send_backup_tasks()

            # Sleep before checking statuses again
            time.sleep(5)

    def _send_backups_tasks(self):
        pass