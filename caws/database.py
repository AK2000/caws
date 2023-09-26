import logging
import threading
from threading import Thread
import queue
import os
import time
import datetime

from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar
X = TypeVar('X')

import sqlalchemy as sa
from sqlalchemy import Column, Text, Float, Boolean, BigInteger, Integer, DateTime, PrimaryKeyConstraint, Table
from sqlalchemy.orm import Mapper
from sqlalchemy.orm import mapperlib
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from caws.task import TaskStatus, CawsTaskInfo

logger = logging.getLogger(__name__)
os.makedirs("logs/", exist_ok=True) 
ch = logging.FileHandler("logs/caws_database.log")
ch.setFormatter(logging.Formatter(
    "[DATABASE]  %(message)s", 'blue'))
logger.addHandler(ch)

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

        self.Session = sessionmaker(bind=self.eng)

        global insert
        if self.eng.dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import insert
        elif self.eng.dialect.name == 'sqlite':
            from sqlalchemy.dialects.sqlite import insert
        else:
            raise NotImplementedError(f"Unsuported dialect for CAWS databse {self.eng.dialect.name}")

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

        with self.Session() as session:
            session.bulk_update_mappings(mapper, mappings)
            session.commit()

    def insert(self, *, table: str, messages: List[dict[str, Any]]) -> None:
        table_obj = self.meta.tables[table]
        mappings = self._generate_mappings(table_obj, messages=messages)
        mapper = self._get_mapper(table_obj)

        with self.Session() as session:
            session.bulk_insert_mappings(mapper, mappings)
            session.commit()

    def insert_or_nothing(self, *, table: str, index_elements: List[str], messages: List[dict[str, Any]]) -> None:
        table_obj = self.meta.tables[table]
        insert_mappings = self._generate_mappings(table_obj, messages=messages)
        stmt = insert(table_obj).values(insert_mappings)
        stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
        with self.Session() as session:
            session.execute(stmt)
            session.commit()

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
        transfer_endpoint_id = Column(Text, nullable=True)
        endpoint_status = Column(Text, nullable=True)
        task_status = Column(Text, nullable=False)
        time_submit = Column(DateTime, nullable=False)
        time_scheduled = Column(DateTime, nullable=True)
        time_began = Column(DateTime, nullable=True)
        time_completed = Column(DateTime, nullable=True)
        running_duration = Column(Float, nullable=True)
        energy_consumed = Column(Float, nullable=True)

    class CawsEndpoint(Base):
        __tablename__ = "caws_endpoint"
        endpoint_id = Column(Text, nullable=False, primary_key=True)
        transfer_endpoint_id = Column(Text, nullable=True)
        tasks_run = Column(Integer, nullable=True)
        static_power = Column(BigInteger, nullable=True)
        energy_consumed = Column(BigInteger, nullable=True)

    class Transfer(Base):
        __tablename__ = "transfer"
        transfer_id = Column(Text, nullable=False, primary_key=True)
        src_endpoint_id = Column(Text, nullable=False)
        dest_endpoint_id = Column(Text, nullable=False)
        transfer_status = Column(Text, nullable=False)
        time_submit = Column(DateTime, nullable=False)
        time_completed = Column(DateTime, nullable=True)
        bytes_transferred = Column(BigInteger, nullable=True)
        files_transferred = Column(Integer, nullable=True)
        sync_level = Column(Integer, nullable=True)

    class TaskFeatures(Base):
        __tablename__ = "features"
        caws_task_id = Column(Text, nullable=False)
        feature_id = Column(Integer, nullable=False)
        feature_type = Column(Text, nullable=False)
        value = Column(Text, nullable=False)

        __table_args__ = (
            PrimaryKeyConstraint('caws_task_id', 'feature_id'),
        )

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CawsDatabaseManager(metaclass=Singleton):
    def __init__(self,
                 db_url,
                 batching_interval : float = 1,
                 batching_threshold: int = 999):
        self.db = CawsDatabase(db_url)
        self.started = False
        self.task_msg_queue = queue.Queue()
        self.transfer_msg_queue = queue.Queue()
        self.feature_msg_queue = queue.Queue()
        self.batching_interval = batching_interval
        self.batching_threshold = batching_threshold

    def start(self):
        if self.started:
            return

        self.started = True
        self._kill_event = threading.Event()
        self._pusher_thread = Thread(target=self._database_pushing_loop)
        self._pusher_thread.start()
        logger.info("Database pusher started")

    def shutdown(self):
        if self.started:
            logger.info("Joining database thread")
            self._kill_event.set()
            self._pusher_thread.join()
            self.started = False

    def update_endpoints(self, msgs):
        self.db.insert_or_nothing(table="caws_endpoint", index_elements=["endpoint_id"], messages=msgs)

    def send_task_message(self, task: CawsTaskInfo):
        msg = dict()
        msg["caws_task_id"] = task.task_id
        if task.gc_future is not None:
            msg["funcx_task_id"] = task.gc_future.task_id
            msg["endpoint_id"] = task.endpoint.compute_endpoint_id
            msg["transfer_endpoint_id"] = task.endpoint.transfer_endpoint_id
            msg["endpoint_status"] = task.endpoint_status.name

        msg["func_name"] = task.function_name
        msg["task_status"] = task.task_status.name
        msg["time_submit"] = task.timing_info.get("submit")
        msg["time_scheduled"] = task.timing_info.get("scheduled")
        msg["time_began"] = task.timing_info.get("began")
        msg["time_completed"] = task.timing_info.get("completed")
        self.task_msg_queue.put(msg)

    def send_transfer_message(self, transfer_info: dict[Any]):
        self.transfer_msg_queue.put(transfer_info)

    def send_feature_message(self, feature_info: dict[Any]):
        self.feature_msg_queue.put(feature_info)

    def _get_messages_in_batch(self, msg_queue: "queue.Queue[X]") -> List[X]:
        messages = []  # type: List[X]
        start = time.time()
        while True:
            if time.time() - start >= self.batching_interval or len(messages) >= self.batching_threshold:
                break
            try:
                x = msg_queue.get(timeout=0.1)
            except queue.Empty:
                break
            else:
                messages.append(x)
        return messages

    def _database_pushing_loop(self):
        task_update_cols = ["funcx_task_id", "endpoint_id", "transfer_endpoint_id", "endpoint_status", "task_status",
                       "time_scheduled", "time_began", "time_completed", "caws_task_id"]
        transfer_update_cols = ["transfer_id", "transfer_status", "time_completed", "bytes_transferred", "files_transferred", "sync_level"]

        while (not self._kill_event.is_set() or
                self.task_msg_queue.qsize() != 0 or self.transfer_msg_queue.qsize() != 0 or
                self.feature_msg_queue.qsize() != 0):
            task_messages = self._get_messages_in_batch(self.task_msg_queue)
            logger.debug(f"Sending {len(task_messages)} task messages to database")
            insert_messages = []
            update_messages = []
            for x in task_messages:
                if x["task_status"] == "CREATED":
                    insert_messages.append(x)
                else:
                    update_messages.append(x)

            if len(insert_messages) > 0:
                self.db.insert(table="caws_task", messages=insert_messages)
            if len(update_messages) > 0:
                self.db.update(table="caws_task", columns=task_update_cols, messages=update_messages)

            transfer_messages = self._get_messages_in_batch(self.transfer_msg_queue)
            logger.debug(f"Sending {len(transfer_messages)} transfer messages to database")
            insert_messages = []
            update_messages = []
            for x in transfer_messages:
                if x["transfer_status"] == "CREATED":
                    insert_messages.append(x)
                else:
                    update_messages.append(x)

            if len(insert_messages) > 0:
                self.db.insert(table="transfer", messages=insert_messages)
            if len(update_messages) > 0:
                self.db.update(table="transfer", columns=transfer_update_cols, messages=update_messages)

            
            feature_messages = self._get_messages_in_batch(self.feature_msg_queue)
            logger.debug(f"Sending {len(feature_messages)} feature messages to database")
            if len(feature_messages) > 0:
                self.db.insert(table="features", messages=feature_messages)
