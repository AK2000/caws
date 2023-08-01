from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence, Optional, TYPE_CHECKING
from concurrent.futures import Future
from uuid import uuid4
from datetime import datetime

from globus_compute_sdk import Client
from globus_compute_sdk.sdk.asynchronous.compute_future import ComputeFuture

from caws.utils import client

if TYPE_CHECKING:
    from caws.endpoint import Endpoint
    from caws.executor import CawsExecutor

class TaskStatus(IntEnum):
    CREATED: int = 0
    WAITING_DEPENDENCIES: int = 1
    READY: int = 2
    SCHEDULED: int = 3
    EXECUTING: int = 4
    COMPLETED: int = 5
    ERROR: int = 6

class CawsFuture(Future):
    task_info: CawsTaskInfo

    def __init__(self, task_info):
        super().__init__()
        self.task_info = task_info

@dataclass
class CawsTaskInfo:
    function_id: str
    task_args: Sequence[Any]
    task_kwargs: dict[str, Any]
    task_id: str
    function_name: str
    task_status: TaskStatus = TaskStatus.CREATED
    transfer_id: int | None = None 
    timing_info: dict[str, datetime.DateTime] = field(default_factory=dict)
    caws_future: Future[Any] | None = None
    gc_future: ComputeFuture[Any] | None = None
    endpoint: Endpoint | None = None

class CawsTask:
    __name__: str
    function_id: str
    func: Callable | None

    def __init__(self, client: Client, func: Optional[Callable] = None, func_id:Optional[str] = None):
        # TODO: Figure out container support
        if func is None and func_id is None:
            raise Exception
        
        if func_id is not None:
            self.function_id = func_id
            self.__name__ = function_id
        else:
            self.func = func
            self.__name__ = func.__name__
            self.function_id = client.register_function(func)

    def __call__(self, *args, **kwargs):
        return func(*args, **kwargs)

def caws_task(function=None):
    def decorator(func):
        return CawsTask(client, func)

    if function is not None:
        return decorator(function)

    return decorator