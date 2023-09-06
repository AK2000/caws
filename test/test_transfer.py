import logging
import time
import os

import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor
from caws.transfer import TransferManager, TransferStatus
from caws.path import CawsPath

from util_funcs import transfer_file

def test_transfer_manager():
    source_path = "test_transfer.txt"
    source_name = "caws-dev_ep1"
    source_compute_id = "27c10959-3ae1-4ce9-a20d-466e7bba293c"
    source_transfer_id = "02ca3fe4-0dee-11ee-bdcc-a3018385fcef"

    dest_name = "caws-dev_ep2"
    dest_compute_id = "ef86d1de-d421-4423-9c6f-09acbfabf5b6"
    dest_transfer_id = "2fde89c0-6fb4-11eb-8c47-0eb1aa8d4337"

    source = caws.Endpoint(source_name,
                           source_compute_id,
                           local_path="/D/UChicago/globus/",
                           transfer_id=source_transfer_id,
                           monitoring_avail=True)

    destination = caws.Endpoint(dest_name,
                                dest_compute_id,
                                transfer_id=dest_transfer_id,
                                monitoring_avail=True)

    def success_callback():
        assert True

    def failure_callback(transfer_record):
        assert False, f"Transfer failed: {transfer_record.error}"

    transfer_manager = TransferManager(log_level=logging.DEBUG)
    transfer_manager.start()
    assert transfer_manager.started

    file_path = CawsPath(source, source_path)
    transfer_record = transfer_manager.transfer([file_path,],
                              destination,
                              "test_transfer",
                              success_callback,
                              failure_callback)

    transfer_manager.shutdown()
    assert not transfer_manager.started
    assert transfer_record.status == TransferStatus.COMPLETED

def test_executor_transfer():
    source_path = "test_transfer.txt"
    source_name = "caws-dev_ep1"
    source_compute_id = "27c10959-3ae1-4ce9-a20d-466e7bba293c"
    source_transfer_id = "02ca3fe4-0dee-11ee-bdcc-a3018385fcef"

    dest_name = "caws-dev_ep2"
    dest_compute_id = "14d17201-7380-4af8-b4e0-192cb9805274"
    dest_transfer_id = "9032dd3a-e841-4687-a163-2720da731b5b"

    source = caws.Endpoint(source_name,
                           source_compute_id,
                           endpoint_path="/D/UChicago/src/research",
                           local_path="/mnt/d/UChicago/src/research",
                           transfer_id=source_transfer_id,
                           monitoring_avail=True)

    destination = caws.Endpoint(dest_name,
                                dest_compute_id,
                                endpoint_path="/alokvk2/",
                                local_path="/home/alokvk2/",
                                transfer_id=dest_transfer_id,
                                monitoring_avail=True)

    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data", "hello_world.txt")
    caws_path = CawsPath(source, file_path)
    endpoints = [destination, source]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy) as executor:
        fut = executor.submit(transfer_file, caws_path)
        assert fut.result()

if __name__ == "__main__":
    test_executor_transfer()