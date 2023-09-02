import sys
import time
import json
import uuid
import logging
import requests
import os
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
from threading import Thread, Lock, Event
from collections import defaultdict
from datetime import datetime

from caws.utils import client
from caws.transfer import TransferManager
from caws.strategy import Strategy
from caws.task import CawsTaskInfo, TaskStatus, CawsFuture
from caws.endpoint import Endpoint, EndpointState
from caws.predictors.transfer_predictors import TransferPredictor
from caws.database import CawsDatabaseManager

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

    def __init__(self, 
                 endpoints: list[Endpoint],
                 strategy: Strategy,
                 caws_database_url: str | None = None,
                 task_watchdog_sleep: float = 0.2,
                 endpoint_watchdog_sleep: float = 5,
                 task_scheduling_sleep: float = 5):
        
        self.endpoints = endpoints
        self.strategy = strategy
        self._transfer_manager = TransferManager(endpoints, log_level=logging.DEBUG)

        self._task_watchdog_sleep = task_watchdog_sleep
        self._endpoint_watchdog_sleep = endpoint_watchdog_sleep
        self._task_scheduling_sleep = task_scheduling_sleep

        if caws_database_url is None:
            caws_database_url = os.environ["ENDPOINT_MONITOR_DEFAULT"]

        self.caws_db = CawsDatabaseManager(caws_database_url)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.shutdown()
        return False

    def start(self):
        print("Executor starting")
        self.caws_db.start()
        self._transfer_manager.start()

        self._kill_event = Event()
        self._task_scheduler = Thread(target=self._schedule_tasks_loop, args=(self._kill_event,))
        self._task_scheduler.start()

        self._endpoint_watchdog = Thread(target=self._check_endpoints, args=(self._kill_event,))
        self._endpoint_watchdog.start()

    def shutdown(self):
        print("Executor shutting down")
        self._kill_event.set()
        self._task_scheduler.join()
        self._endpoint_watchdog.join()
        self._transfer_manager.shutdown()
        self.caws_db.shutdown()

    def submit(self, fn: Callable, *args, **kwargs):
        task_id = str(uuid.uuid4())
        task_info = CawsTaskInfo(fn, args, kwargs, task_id, fn.__name__)
        task_info.caws_future = CawsFuture(task_info)
        task_info.timing_info["submit"] = datetime.now()
        self.caws_db.send_monitoring_message(task_info)
        
        with self.ready_tasks_lock:
            self.ready_tasks.append(task_info)

        return task_info.caws_future

    def _schedule_tasks_loop(self, kill_event):
        print("Starting task-submission thread")

        self.tasks_scheduling = []
        while not kill_event.is_set():
            with self.ready_tasks_lock:
                self.tasks_scheduling.extend(self.ready_tasks)
                self.ready_tasks = []
            
            scheduling_decisions, self.tasks_scheduling = self.strategy.schedule(self.tasks_scheduling)

            for task, endpoint in scheduling_decisions:
                print("Scheduling tasks")
                self._schedule_task(task, endpoint)

            if len(scheduling_decisions) == 0:
                time.sleep(self._task_scheduling_sleep)

    def _schedule_task(self, task, endpoint):
        print("In scheduled task")
        task.task_status = TaskStatus.SCHEDULED
        task.timing_info["scheduled"] = datetime.now()
        print("Submitting message to db")
        # self.caws_db.send_monitoring_message(task_info)

        print("Reading files")
        files = task.task_kwargs.get("_globus_files", {}) # Must be dict of {src: [<paths>]}
        print(files)

        # Start Globus transfer of required files, if any
        if len(files) > 0:
            print(f"Starting file transfers for task {task.task_id}")
            task.transfer_record = self._transfer_manager.transfer(files, 
                                                                  endpoint,
                                                                  task.task_id, 
                                                                  callback= lambda : self._start_task(task, endpoint))
            endpoint.schedule(task)
        else:
            print("Staring task")
            self._start_task(task, endpoint)

    def _start_task(self, task, endpoint):
        task.task_status = TaskStatus.EXECUTING
        task.endpoint_status = endpoint.state
        task.timing_info["began"] = datetime.now()
        
        print("Submitting task to endpoint")
        endpoint.submit(task)
        self.caws_db.send_monitoring_message(task)

        # Must be done last to avoid race condition
        task.gc_future.add_done_callback(lambda fut : self._task_complete_callback(task, fut))

    def _task_complete_callback(self, task, fut):
        if fut.exception() is not None:
            task.task_status = TaskStatus.ERROR
        else:
            task.task_status = TaskStatus.COMPLETED

        task.timing_info["completed"] = datetime.now()
        self.caws_db.send_monitoring_message(task)
        task.endpoint.task_finished(task)

        if fut.exception() is not None:
            task.caws_future.set_exception(fut.exception())
        else:
            task.caws_future.set_result(fut.result())

    def _check_endpoints(self, kill_event):
        logger.info('Starting endpoint-watchdog thread')

        while not kill_event.is_set():
            for endpoint in self.endpoints:
                state = endpoint.poll()
                if state == EndpointState.DEAD:
                    self._send_backup_tasks()

            # Sleep before checking statuses again
            time.sleep(5)

    def _send_backup_tasks(self):
        pass