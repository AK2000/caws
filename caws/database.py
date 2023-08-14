import logging
import threading
from threading import Thread
import queue
import os
import time
import datetime

from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, cast

import sqlalchemy as sa
from sqlalchemy import Column, Text, Float, Boolean, BigInteger, Integer, DateTime, PrimaryKeyConstraint, Table
from sqlalchemy.orm import Mapper
from sqlalchemy.orm import mapperlib
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from caws.task import TaskStatus, CawsTaskInfo

class CawsDatabase:

    Base = declarative_base()

    def __init__(self,
                 url: str,
                 ):
        self.eng = sa.create_engine(url)
        self.meta = self.Base.metadata

        # TODO: this code wants a read lock on the sqlite3 database, and fails if it cannot
        # - for example, if someone else is querying the database at the point that the
        # monitoring system is initialized. See PR #1917 for related locked-for-read fixes
        # elsewhere in this file.
        self.meta.create_all(self.eng)

        self.meta.reflect(bind=self.eng)

        Session = sessionmaker(bind=self.eng)
        self.session = Session()

    def _get_mapper(self, table_obj: Table) -> Mapper:
        all_mappers: Set[Mapper] = set()
        for mapper_registry in mapperlib._all_registries():  # type: ignore
            all_mappers.update(mapper_registry.mappers)
        mapper_gen = (
            mapper for mapper in all_mappers
            if table_obj in mapper.tables
        )
        try:
            mapper = next(mapper_gen)
            second_mapper = next(mapper_gen, False)
        except StopIteration:
            raise ValueError(f"Could not get mapper for table {table_obj}")

        if second_mapper:
            raise ValueError(f"Multiple mappers for table {table_obj}")
        return mapper

    def update(self, *, table: str, columns: List[str], messages: List[dict[str, Any]]) -> None:
        table_obj = self.meta.tables[table]
        mappings = self._generate_mappings(table_obj, columns=columns,
                                           messages=messages)
        mapper = self._get_mapper(table_obj)
        self.session.bulk_update_mappings(mapper, mappings)
        self.session.commit()

    def insert(self, *, table: str, messages: List[dict[str, Any]]) -> None:
        table_obj = self.meta.tables[table]
        mappings = self._generate_mappings(table_obj, messages=messages)
        mapper = self._get_mapper(table_obj)
        self.session.bulk_insert_mappings(mapper, mappings)
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def _generate_mappings(self, table: Table, columns: Optional[List[str]] = None, messages: List[Dict[str, Any]] = []) -> List[Dict[str, Any]]:
        mappings = []
        for msg in messages:
            m = {}
            if columns is None:
                columns = table.c.keys()
            for column in columns:
                m[column] = msg.get(column, None)
            mappings.append(m)
        return mappings

    class CawsTask(Base):
        __tablename__ = "caws_task"
        caws_task_id = Column(Text, nullable=False, primary_key=True)
        funcx_task_id = Column(Text, nullable=True)
        func_name = Column(Text, nullable=False)
        endpoint_id = Column(Text, nullable=True)
        task_status = Column(Text, nullable=False)
        time_submit = Column(DateTime, nullable=False)
        time_scheduled = Column(DateTime, nullable=True)
        time_began = Column(DateTime, nullable=True)
        time_completed = Column(DateTime, nullable=True)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CawsDatabaseManager(metaclass=Singleton):
    def __init__(self,
                 db_url,
                 batching_interval : float = 2,
                 batching_threshold: int = 10):
        self.db = CawsDatabase(db_url)
        self.started = False
        self.msg_queue = queue.Queue()
        self.batching_interval = batching_interval
        self.batching_threshold = batching_threshold

    def start(self):
        if self.started:
            return

        self.started = True
        self._kill_event = threading.Event()
        self._pusher_thread = Thread(target=self._database_pushing_loop)
        self._pusher_thread.start()
        print("Database pusher started")

    def shutdown(self):
        if self.started:
            self._kill_event.set()
            print("Joining database thread")
            self._pusher_thread.join()
            self.started = False

    def send_monitoring_message(self, task: CawsTaskInfo):
        msg = dict()
        msg["caws_task_id"] = task.task_id
        if task.gc_future is not None:
            msg["funcx_task_id"] = task.gc_future.task_id
        msg["func_name"] = task.function_name
        msg["task_status"] = task.task_status.name
        msg["time_submit"] = task.timing_info.get("submit")
        msg["time_scheduled"] = task.timing_info.get("scheduled")
        msg["time_began"] = task.timing_info.get("began")
        msg["time_completed"] = task.timing_info.get("completed")

        self.msg_queue.put(msg)

    def _database_pushing_loop(self):
        update_cols = ["funcx_task_id", "endpoint_id", "task_status", "time_scheduled", "time_began", "time_completed", "caws_task_id"]
        while not (self._kill_event.is_set() and self.msg_queue.empty()):
            num_msgs = 0
            insert_messages = []
            update_messages = []

            start = time.time()
            while True:
                if time.time() - start >= self.batching_interval or num_msgs >= self.batching_threshold:
                    break

                try:
                    remaining = max(0, self.batching_interval - (time.time() - start))
                    x = self.msg_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                else:
                    num_msgs += 1
                    if x["task_status"] == "CREATED":
                        insert_messages.append(x)
                    else:
                        update_messages.append(x)

            if len(insert_messages) > 0:
            # TODO: Surround in try, except, retry, backoff
                self.db.insert(table="caws_task", messages=insert_messages)
            
            if len(update_messages) > 0:
                self.db.update(table="caws_task", columns=update_cols, messages=update_messages)