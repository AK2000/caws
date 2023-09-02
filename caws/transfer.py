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
    error: None | str = None

class TransferManager(object):
    def __init__(self, endpoints, sync_level='exists', log_level=logging.INFO):
        # Authorize with globus
        self.authorize()

        self.transfer_client = globus_sdk.TransferClient(authorizer=self.authorizer)
        self.endpoints = endpoints
        self.name_to_endpoints = {}
        for endpoint in self.endpoints:
            self.name_to_endpoints[endpoint.name] = endpoint
        self.endpoints = endpoints
        self.sync_level = sync_level
        logger.setLevel(log_level)

        # Track pending transfers
        self._next = 0
        self.active_transfers = {}

        self._polling_interval = 1
        self.started = False

    def authorize(self):
        # this is the tutorial client ID
        CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"
        client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

        client.oauth2_start_flow(refresh_tokens=True)
        authorize_url = client.oauth2_get_authorize_url()
        print(f"Please go to this URL and login:\n\n{authorize_url}\n")

        auth_code = input("Please enter the code you get after login here: ").strip()
        token_response = client.oauth2_exchange_code_for_tokens(auth_code)

        globus_transfer_data = token_response.by_resource_server["transfer.api.globus.org"]
        transfer_rt = globus_transfer_data["refresh_token"]
        transfer_at = globus_transfer_data["access_token"]
        expires_at_s = globus_transfer_data["expires_at_seconds"]

        # construct a RefreshTokenAuthorizer
        # note that `client` is passed to it, to allow it to do the refreshes
        self.authorizer = globus_sdk.RefreshTokenAuthorizer(
            transfer_rt, client, access_token=transfer_at, expires_at=expires_at_s
        )

    def start(self):
        if self.started:
            return

        self.started = True
        self._terminate_event = Event()

        # Initialize thread to wait on transfers
        self._tracker = Thread(target=self._track_transfers, daemon=True)
        self._tracker.start()
    
    def shutdown(self):
        if not self.started:
            return
        
        self._terminate_event.set()
        self._tracker.join()
        self.started = False

    def transfer(self, 
                 files_by_src, 
                 dst,
                 task_name="",
                 unique_name=False,
                 callback=None,
                 failed_callback=None):

        task_record = TransferRecord(self._next, [], 0, TransferStatus.CREATING, callback, failed_callback)
        self._next += 1 #TODO: Do I need to worry about thread safety here?

        n = len(files_by_src)
        for i, (src_name, files) in enumerate(files_by_src.items(), 1):
            src = self.name_to_endpoints[src_name]
            dst_name = dst.name

            if src.transfer_endpoint_id == dst.transfer_endpoint_id:
                logger.debug(f'Skipped transfer from {src_name} to {dst_name}')
                continue

            # files, _ = zip(*pairs)
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
                'name': f"{task_name} {i}/{n}",
                'src_id': src.transfer_endpoint_id,
                'dest_id': dst.transfer_endpoint_id,
                'files': files,
                'submission_time': time.time(),
                'task_record': task_record
            }
            task_record.transfer_ids.append(res['task_id'])
            logger.debug(f"Submited Transfer to Globus, task id: {res['task_id']}")
        
        task_record.status = TransferStatus.PENDING
        if task_record.remaining == 0:
            if not task_record.error:
                if task_record.callback is not None:
                    task_record.callback()
                task_record.status = TransferStatus.COMPLETED
            else:
                if task_record.failed_callback is not None:
                    task_record.failed_callback()
                task_record.status = TransferStatus.FAILED
        else:
            task_record.status = TransferStatus.EXECUTING
            
        return task_record

    def _update_transfer_record(self, task_record, status):
        if status['status'] == 'FAILED':
            task_record.error = status['fatal_error']['description']

        task_record.remaining -= 1
        if task_record.status == TransferStatus.CREATING:
            return

        while task_record.status == TransferStatus.PENDING:
            time.sleep(0)
        
        if task_record.status != TransferStatus.EXECUTING:
            return

        logger.debug("Transfer task currently executing")
        if task_record.remaining == 0:
            if task_record.error:
                logger.debug("Transfer failed")
                task_record.status = TransferStatus.FAILED
                task_record.failed_callback(task_record)
            else:
                # TODO: Should we fail as soon as possible?
                logger.debug("Transfer completed successfully")
                task_record.status = TransferStatus.COMPLETED
                logger.debug("Executing callback")
                task_record.callback()

    def _track_transfers(self):
        logger.info('Started transfer tracking thread')

        next_send = time.time() + self._polling_interval
        while (not self._terminate_event.is_set()) or (len(self.active_transfers) != 0):
            for transfer_id, info in list(self.active_transfers.items()):
                name = info['name']
                logger.debug(f"Polling transfer {name}")
                status = self.transfer_client.get_task(transfer_id)

                if status['status'] == 'ACTIVE':
                    continue

                if status['status'] == 'FAILED':
                    logger.error('Task {} failed. Canceling task!'
                                 .format(transfer_id))
                    res = self.transfer_client.cancel_task(transfer_id)
                    if res['code'] != 'Canceled':
                        logger.error('Could not cancel task {}. Reason: {}'
                                     .format(transfer_id, res['message']))

                elif status['status'] == 'SUCCEEDED':
                    info['time_taken'] = time.time() - info['submission_time']
                    logger.info('Globus transfer {} finished in time {}'
                                .format(name, info['time_taken']))

                self._update_transfer_record(info["task_record"], status)
                del self.active_transfers[transfer_id]

            time.sleep(max(0, next_send - time.time()))
            next_send += self._polling_interval
        
        logger.info('Ending transfer tracking thread')