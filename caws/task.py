from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence, Optional
from concurrent.futures import Future
from uuid import uuid4

from globus_compute_sdk import Client

class TaskStatus(Enum):
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
    task_status: TaskStatus = TaskStatus.CREATED
    transfer_id: int | None = None 
    timing: dict[str, int] = field(default_factory=dict)
    caws_future: Future[Any] | None = None
    gc_future: Future[Any] | None = None

    def _update_caws_future(self, fut):
        if fut.exception() is not None:
            self.caws_future.set_exception(fut.exception())
        else:
            self.caws_future.set_result(fut.result())

class CawsTask:
    function_id: str
    func: Callable | None

    def __init__(self, client: Client, func: Optional[Callable] = None, func_id:Optional[str] = None):
        # TODO: Figure out container support
        if func is None and func_id is None:
            raise Exception
        
        if func is not None:
            self.func = func
            self.function_id = client.register_function(func)

    def __call__(self, *args, **kwargs):
        return func(*args, **kwargs)

def caws_task(function=None):
    from caws.executor import client

    def decorator(func):
        return CawsTask(client, func)

    if function is not None:
        return decorator(function)

    return decorator