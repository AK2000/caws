import os
import uuid
import time
import logging
import threading
from threading import Thread, Event
import json
import numpy as np
from queue import Queue
from enum import IntEnum
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import globus_sdk

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    "[TRANSFER]  %(message)s", 'red'))
logger.addHandler(ch)

class TransferStatus(IntEnum):
    CREATING: int = 0
    PENDING: int = 1
    EXECUTING: int = 2
    COMPLETED: int = 3
    ERROR: int = 4

@dataclass
class TransferRecord:
    task_id: int
    transfer_ids: list[str]
    remaining: int
    status: TransferStatus
    callback: None | Callable = None
    fail_callback: None | Callable = None
    callback_thread: None | int = None
    error: int | None = None

class TransferManager(object):
    def __init__(self, endpoints, sync_level='exists', log_level='INFO'):
        self.transfer_client = globus_sdk.TransferClient() #TODO: implement authorizer
        self.endpoints = endpoints
        self.name_to_endpoints = {}
        for endpoint in self.endpoints:
            self.name_to_endpoints[endpoint.name] = endpoints
        self.endpoints = endpoints
        self.sync_level = sync_level
        logger.setLevel(log_level)

        # Track pending transfers
        self._next = 0
        self.active_transfers = {}
        self.completed_transfers = {}
        self.transfers_to_tasks = {}
        self.task_records = {}

        self._polling_interval = 1
        self._started = False

    def start(self):
        if self.started:
            return

        self.started = True
        self._terminate_event = Event()

        # Initialize thread to wait on transfers
        self._tracker = Thread(target=self._track_transfers)
        self._tracker.daemon = True
        self._tracker.start()
    
    def shutdown(self):
        if not self.started:
            return
        
        self._terminate_event.set()

    def transfer(self, 
                 files_by_src, 
                 dst, 
                 task_id='', 
                 unique_name=False,
                 callback=None,
                 failed_callback=None):

        task_record = TransferRecord(self._next, [], 0, TransferStatus.CREATING, callback, fail_callback)
        self.task_records[task_record.task_id] = task_record 
        self._next += 1 #TODO: Do I need to worry about thread safety here?

        n = len(files_by_src)
        for i, (src_name, pairs) in enumerate(files_by_src.items(), 1):
            src = self.name_to_endpoints[src_name]
            dst_name = dst.name

            if src.transfer_endpoint_id == dst.transfer_endpoint_id:
                logger.debug(f'Skipped transfer from {src_name} to {dst_name}')
                continue

            files, _ = zip(*pairs)
            logger.info(f'Transferring {src_name} to {dst_name}: {files}')

            tdata = globus_sdk.TransferData(self.transfer_client,
                                            src.transfer_endpoint_id,
                                            dst.transfer_endpoint_id,
                                            label='FuncX Transfer {} - {} of {}'
                                            .format(self._next + 1, i, n),
                                            sync_level=self.sync_level)

            for f in files:
                if unique_name:
                    dst_file = '~/.globus_funcx/test_{}.txt'.format(
                        str(uuid.uuid4()))
                    logger.debug('Unique destination file name: {}'
                                 .format(dst_file))
                    tdata.add_item(f, dst_file)
                else:
                    tdata.add_item(f, f)

            res = self.transfer_client.submit_transfer(tdata)

            if res['code'] != 'Accepted':
                raise ValueError('Transfer not accepted')

            task_record.remaining += 1
            self.active_transfers[res['task_id']] = {
                'src': src_globus,
                'dst': dst_globus,
                'files': files,
                'name': f'{task_id} ({i}/{n})',
                'submission_time': time.time()
            }
            task_record.transfer_ids.append(res['task_id'])
        
        task_record.status = PENDING
        if task_record.remaining == 0:
            if not task_record.error:
                task_record.callback()
                task_record.status = TransferStatus.COMPLETED
            else:
                task_record.failed_callback()
                task_record.status = TransferStatus.FAILED
        else:
            task_record.status = TransferStatus.EXECUTING
            
        return task_record
        


    def _track_transfers(self):
        logger.info('Started transfer tracking thread')

        next_send = time.time() + self._polling_interval
        while not self._terminate_event.is_set():
            for transfer_id, info in list(self.active_transfers.items()):
                name = info['name']
                status = self.transfer_client.get_task(transfer_id)

                if status['status'] == 'FAILED':
                    logger.error('Task {} failed. Canceling task!'
                                 .format(transfer_id))
                    res = self.transfer_client.cancel_task(transfer_id)
                    if res['code'] != 'Canceled':
                        logger.error('Could not cancel task {}. Reason: {}'
                                     .format(transfer_id, res['message']))
                    del self.active_transfers[transfer_id]

                elif status['status'] == 'ACTIVE':
                    continue

                elif status['status'] == 'SUCCEEDED':
                    info['time_taken'] = time.time() - info['submission_time']
                    logger.info('Globus transfer {} finished in time {}'
                                .format(name, info['time_taken']))
                    self.completed_transfers[transfer_id] = info
                    del self.active_transfers[transfer_id]

        self._terminate_event.wait(max(0, next_send - time.time()))
        next_send += self._polling_interval