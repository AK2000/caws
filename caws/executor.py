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
import inspect

from caws.utils import create_output_dir_wrapper
from caws.transfer import TransferManager, TransferException
from caws.strategy.base import Strategy
from caws.task import CawsTaskInfo, TaskStatus, CawsFuture, CawsTask
from caws.endpoint import Endpoint, EndpointState
from caws.predictors.transfer_predictors import TransferPredictor
from caws.database import CawsDatabaseManager
from caws.path import CawsPath

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
    _endpoint_watchdog_sleep: float = 1
    _task_scheduling_sleep: float = 0.5

    scheduling_lock = Lock() # Use to manually batch tasks
    ready_tasks_lock = Lock()
    ready_tasks: list[CawsTaskInfo] = []

    def __init__(self, 
                 endpoints: list[Endpoint],
                 strategy: Strategy,
                 predictor = None,
                 caws_database_url: str | None = None,
                 task_watchdog_sleep: float = 0.2,
                 endpoint_watchdog_sleep: float = 1):
        
        self.endpoints = endpoints
        self.strategy = strategy
        self.predictor = predictor
        self._task_watchdog_sleep = task_watchdog_sleep
        self._endpoint_watchdog_sleep = endpoint_watchdog_sleep

        if caws_database_url is None:
            caws_database_url = os.environ["ENDPOINT_MONITOR_DEFAULT"]
        self.caws_db = CawsDatabaseManager(caws_database_url)

        self._transfer_manager = TransferManager(caws_db=self.caws_db, log_level=logging.DEBUG)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        self.shutdown()
        return False

    def start(self):
        print("Executor starting")
        msgs = []
        for endpoint in self.endpoints:
            msg = {
                "endpoint_id": endpoint.compute_endpoint_id,
                "transfer_endpoint_id": endpoint.transfer_endpoint_id,
                "tasks_run": 0,
                "energy_consumed": 0,
            }
            msgs.append(msg)
        self.caws_db.update_endpoints(msgs)

        if self.predictor:
            self.predictor.start()
        
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
        self._transfer_manager.shutdown()
        self.caws_db.shutdown()

        if self.predictor:
            time.sleep(2)
            self.predictor.update()

        self._task_scheduler.join()
        self._endpoint_watchdog.join()
        

    def submit(self, fn: Callable, *args, deadline=None, resources=None, **kwargs):
        if isinstance(fn, CawsTask):
            features = fn.extract_features(*args, **kwargs)
            fn = fn.extract_func()
        else:
            features = []

        task_id = str(uuid.uuid4())
        name = kwargs.get("name", f"{fn.__module__}.{fn.__qualname__}")
        task_info = CawsTaskInfo(fn, list(args), kwargs, task_id, name, features=features)
        task_info.caws_future = CawsFuture(task_info)
        task_info.timing_info["submit"] = datetime.now()
        self.caws_db.send_task_message(task_info)

        for i, (value, feature_type) in enumerate(features):
            self.caws_db.send_feature_message({
                    "caws_task_id": task_id,
                    "feature_id": i,
                    "feature_type": feature_type.name,
                    "value": str(value)
            })
        
        with self.ready_tasks_lock:
            self.ready_tasks.append(task_info)

        return task_info.caws_future

    def _schedule_tasks_loop(self, kill_event):
        print("Starting task-submission thread")

        self.tasks_scheduling = []
        while not kill_event.is_set():
            with self.scheduling_lock:
                with self.ready_tasks_lock:
                    self.tasks_scheduling.extend(self.ready_tasks)
                    self.ready_tasks = []
            
            scheduling_decisions, self.tasks_scheduling = self.strategy.schedule(self.tasks_scheduling)

            for task, endpoint in scheduling_decisions:
                print("Scheduling tasks")
                self._schedule_task(task, endpoint)
            self._transfer_manager.submit_pending_transfers()

            if len(scheduling_decisions) == 0:
                time.sleep(self._task_scheduling_sleep)

    def _schedule_task(self, task, endpoint):
        task.task_status = TaskStatus.SCHEDULED
        task.timing_info["scheduled"] = datetime.now()
        self.caws_db.send_task_message(task)

        # Replacing input files with paths after Globus transfers
        files = []
        for i, arg in enumerate(task.task_args):
            if isinstance(arg, CawsPath):
                files.append(arg)
                task.task_args[i] = arg.get_dest_local_path(endpoint, str(task.task_id))

        for key, arg in task.task_kwargs.items():
            if isinstance(arg, CawsPath):
                files.append(arg)
                task.task_kwargs[key] = arg.get_dest_local_path(endpoint, str(task.task_id))

        # Provide function endpoint path for storing results
        # This way we provide a mechanism so task outputs don't conflict
        # with each other and can be written to a Globus accessible directory
        # TODO: Figure out how this might work with images and K8s
        # TODO: Figure out how convert returned paths into CawsPaths. Maybe register a deserializer?
        try:
            sig = inspect.signature(task.func)
            if "_caws_output_dir" in sig.parameters:
                task.task_kwargs["_caws_output_dir"] = os.path.join(endpoint.local_path, ".caws", str(task.task_id))
                # task.func = create_output_dir_wrapper(task.func) # Create output dir
        except:
            pass

        # Start Globus transfer of required files, if any
        if len(files) > 0:
            print(f"Starting file transfers for task {task.task_id}")
            endpoint.schedule(task)
            task.transfer_record = self._transfer_manager.transfer(files, 
                                                                  endpoint,
                                                                  str(task.task_id), 
                                                                  callback = lambda : self._start_task(task, endpoint),
                                                                  failed_callback = lambda : self._transfer_error(task, endpoint))
        else:
            print("Staring task")
            self._start_task(task, endpoint)

    def _start_task(self, task, endpoint):
        task.task_status = TaskStatus.EXECUTING
        task.endpoint_status = endpoint.state
        task.timing_info["began"] = datetime.now()
        
        print("Submitting task to endpoint")
        endpoint.submit(task)
        self.caws_db.send_task_message(task)

        # Must be done last to avoid race condition
        task.gc_future.add_done_callback(lambda fut : self._task_complete_callback(task, fut))

    def _transfer_error(self, task, endpoint):
        task.task_status = TaskStatus.ERROR
        task.timing_info["completed"] = datetime.now()
        self.caws_db.send_task_message(task)
        endpoint.discard(task)
        task.caws_future.set_exception(TransferException(task.transfer_record.error))

    def _task_complete_callback(self, task, fut):
        if fut.exception() is not None:
            task.task_status = TaskStatus.ERROR
        else:
            task.task_status = TaskStatus.COMPLETED

        task.timing_info["completed"] = datetime.now()
        self.caws_db.send_task_message(task)
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
            time.sleep(self._endpoint_watchdog_sleep)

    def _send_backup_tasks(self):
        pass
