import os
import uuid
import time
from datetime import datetime
import logging
import threading
from threading import Thread, Event
import json
import numpy as np
from queue import Queue
from enum import IntEnum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional
import shutil
import json

import globus_sdk
from globus_sdk.scopes import TransferScopes
from globus_sdk.tokenstorage import SimpleJSONFileAdapter

from caws.database import CawsDatabaseManager

logger = logging.getLogger(__name__)
os.makedirs("logs/", exist_ok=True) 
ch = logging.FileHandler("logs/caws_transfer.log")
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
    remaining: int
    status: TransferStatus
    transfer_ids: list = field(default_factory=list)
    callback: None | Callable = None
    fail_callback: None | Callable = None
    error: None | str = None

class TransferManager(object):
    def __init__(self, caws_db: Optional[CawsDatabaseManager] = None, sync_level=None, log_level=logging.INFO):
        # Authorize with globus
        self.transfer_client = self.get_transfer_client()
        self.sync_level = sync_level
        logger.setLevel(log_level)

        # Track pending transfers
        self._next = 0
        self._round = 0
        self.pending_transfers = {}
        self.active_transfers = {}

        self._polling_interval = 1
        self.started = False
        self.caws_db = caws_db

    def login_and_get_transfer_client(self, *, scopes=TransferScopes.all):
        # note that 'requested_scopes' can be a single scope or a list
        # this did not matter in previous examples but will be leveraged in
        # this one
        self.auth_client.oauth2_start_flow(refresh_tokens=True, requested_scopes=scopes)
        authorize_url = self.auth_client.oauth2_get_authorize_url()
        print(f"Please go to this URL and login:\n\n{authorize_url}\n")

        auth_code = input("Please enter the code here: ").strip()
        token_response = self.auth_client.oauth2_exchange_code_for_tokens(auth_code)
        self.token_file.store(token_response)
        tokens = token_response.by_resource_server["transfer.api.globus.org"]

        authorizer = globus_sdk.RefreshTokenAuthorizer(
                tokens["refresh_token"],
                self.auth_client,
                access_token=tokens["access_token"],
                expires_at=tokens["expires_at_seconds"],
                on_refresh=self.token_file.on_refresh
            )
        # return the TransferClient object, as the result of doing a login
        return globus_sdk.TransferClient(
            authorizer=authorizer
        )

    def get_transfer_client(self):
        # this is the tutorial client ID
        CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"
        self.auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
        self.scopes = [TransferScopes.all]
        self.token_file = SimpleJSONFileAdapter(os.path.expanduser("~/.caws/globus_tokens.json"))

        if not self.token_file.file_exists():
            return self.login_and_get_transfer_client()
        else:
            tokens = self.token_file.get_token_data("transfer.api.globus.org")
            authorizer = globus_sdk.RefreshTokenAuthorizer(
                tokens["refresh_token"],
                self.auth_client,
                access_token=tokens["access_token"],
                expires_at=tokens["expires_at_seconds"],
                on_refresh=self.token_file.on_refresh
            )

            return globus_sdk.TransferClient(
                authorizer=authorizer
            )                      

    def start(self):
        if self.started:
            return

        self.started = True
        self._terminate_event = Event()

        # Initialize thread to wait on transfers
        self._tracker = Thread(target=self._track_transfers)
        self._tracker.start()
    
    def shutdown(self):
        if not self.started:
            return
        
        self._terminate_event.set()
        self._tracker.join()
        self.started = False

    def create_transfer(self, src_endpoint_id, dst_endpoint_id):
        tdata = globus_sdk.TransferData(source_endpoint=src_endpoint_id,
                                            destination_endpoint=dst_endpoint_id,
                                            label='Caws Transfer {}'
                                            .format(self._round),
                                            sync_level=self.sync_level)
        return tdata
        

    def transfer(self, 
                 files, 
                 dst,
                 task_id="",
                 callback=None,
                 failed_callback=None):

        task_record = TransferRecord(self._next, 0, TransferStatus.CREATING, callback=callback, fail_callback=failed_callback)
        self._next += 1 #TODO: Do I need to worry about thread safety here?

        files_by_src = defaultdict(list)
        for src_path in files:
            files_by_src[src_path.endpoint].append(src_path)

        n = len(files_by_src)
        for i, (src, files) in enumerate(files_by_src.items()):
            src_name = src.name
            dst_name = dst.name

            if src.transfer_endpoint_id == dst.transfer_endpoint_id:
                logger.debug(f'Skipped transfer from {src_name} to {dst_name}')
                for src_path in files:
                    dest_path = src_path.get_dest_local_path(dst, task_id)
                    dir_name = os.path.dirname(dest_path)
                    os.makedirs(dir_name, exist_ok=True)
                    if os.path.isfile(src_path.get_src_local_path()):
                        shutil.copy2(src_path.get_src_local_path(),  dest_path)
                    else:
                        shutil.copytree(src_path.get_src_local_path(), dest_path)

                continue

            logger.info(f'Transferring {src_name} to {dst_name}: {files}')

            if (src.transfer_endpoint_id, dst.transfer_endpoint_id) not in self.pending_transfers:
                tdata = self.create_transfer(src.transfer_endpoint_id, dst.transfer_endpoint_id)
                self.pending_transfers[(src.transfer_endpoint_id, dst.transfer_endpoint_id)] = (tdata, [task_record])
            else:
                tdata, _ = self.pending_transfers[(src.transfer_endpoint_id, dst.transfer_endpoint_id)]
                self.pending_transfers[(src.transfer_endpoint_id, dst.transfer_endpoint_id)][1].append(task_record)
            
            size = 0
            for src_path in files:
                size += src_path.size
                tdata.add_item(src_path.get_src_endpoint_path(), src_path.get_dest_endpoint_path(dst, task_id))
                if not src_path.isolate:
                    tdata.sync_level = "exists"
                    
            task_record.remaining += 1

        if task_record.remaining == 0:
            if task_record.callback is not None:
                task_record.callback()
            task_record.status = TransferStatus.COMPLETED
        return task_record

    def submit_pending_transfers(self):
        all_task_records = []
        n = len(self.pending_transfers)
        for i, ((src_endpoint_id, dst_endpoint_id), (tdata, task_records)) in enumerate(self.pending_transfers.items(), start=1):
            try:
                res = self.transfer_client.submit_transfer(tdata)
            except globus_sdk.TransferAPIError as err:
                if err.info.consent_required:
                    print(
                        "Got a ConsentRequired error with scopes:",
                        err.info.consent_required.required_scopes,
                    )
                    print("You will have to login with Globus again")
                    self.scopes.extend(err.info.consent_required.required_scopes)
                    self.transfer_client = self.login_and_get_transfer_client(scopes=self.scopes)
                    res = self.transfer_client.submit_transfer(tdata)
                else:
                    print(err)
                    for task_record in task_records:
                        task_record.status = TransferStatus.PENDING
                        if task_record.fail_callback is not None:
                            task_record.fail_callback()
                        task_record.status = TransferStatus.FAILED
                    continue
                    
            if res['code'] != 'Accepted':
                for task_record in task_records:
                    task_record.status = TransferStatus.PENDING
                    if task_record.failed_callback is not None:
                        task_record.failed_callback()
                    task_record.status = TransferStatus.FAILED
                continue

            
            self.active_transfers[res['task_id']] = {
                'name': f"{self._round} {i}/{n}",
                'transfer_id': res["task_id"],
                'src_endpoint_id': src_endpoint_id,
                'dest_endpoint_id': dst_endpoint_id,
                'transfer_status': "CREATED",
                'time_submit': datetime.now(),
                'task_records': task_records
            }
            logger.debug(f"Submited Transfer to Globus, task id: {res['task_id']}")

            for task_record in task_records:
                task_record.transfer_ids.append(res['task_id'])

            all_task_records.extend(task_records)

            if self.caws_db:
                self.caws_db.send_transfer_message(self.active_transfers[res['task_id']])

        for task_record in all_task_records:
            # Ensures extactly once semantics for callback
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

        self.pending_transfers = {}
        self._round += 1

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
                # TODO: Should we fail as soon as possible?
                logger.debug("Transfer failed")
                task_record.status = TransferStatus.FAILED
                task_record.failed_callback(task_record)
            else:
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

                info['time_completed'] = datetime.now()
                if status['status'] == 'FAILED':
                    info["transfer_status"] = "FAILED"
                    logger.error('Task {} failed. Canceling task!'
                                 .format(transfer_id))
                    res = self.transfer_client.cancel_task(transfer_id)
                    if res['code'] != 'Canceled':
                        logger.error('Could not cancel task {}. Reason: {}'
                                     .format(transfer_id, res['message']))

                elif status['status'] == 'SUCCEEDED':
                    info["transfer_status"] = "SUCCEEDED"
                    info["bytes_transferred"] = status["bytes_transferred"]
                    info["sync_level"] = status["sync_level"]
                    info["files_transferred"] = status["files_transferred"]
                    
                    logger.info('Globus transfer {} finished in time {}'
                                .format(name, info['time_completed'] - info['time_submit']))

                for task_record in info["task_records"]:
                    self._update_transfer_record(task_record, status)

                if self.caws_db:
                    self.caws_db.send_transfer_message(info)
                del self.active_transfers[transfer_id]

            time.sleep(max(0, next_send - time.time()))
            next_send += self._polling_interval
        
        logger.info('Ending transfer tracking thread')

class TransferException(Exception):
    pass
