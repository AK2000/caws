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
    from caws.endpoint import Endpoint, EndpointState
    from caws.executor import CawsExecutor
    from caws.transfer import TransferRecord

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
    func: Callable
    task_args: Sequence[Any]
    task_kwargs: dict[str, Any]
    task_id: str
    function_name: str
    task_status: TaskStatus = TaskStatus.CREATED
    transfer_record: TransferRecord | None = None 
    timing_info: dict[str, datetime.DateTime] = field(default_factory=dict)
    caws_future: Future[Any] | None = None
    gc_future: ComputeFuture[Any] | None = None
    endpoint: Endpoint | None = None
    endpoint_status: EndpointState | None = None
    deadline: datetime.Datetime | None = None