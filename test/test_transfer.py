import logging
import time

import caws
from caws.transfer import TransferManager, TransferStatus

def test_transfer_manager():
    source_path = "/D/UChicago/globus/test_transfer.txt"
    source_name = "caws-dev_ep1"
    source_compute_id = "27c10959-3ae1-4ce9-a20d-466e7bba293c"
    source_transfer_id = "02ca3fe4-0dee-11ee-bdcc-a3018385fcef"

    dest_name = "caws-dev_ep2"
    dest_compute_id = "ef86d1de-d421-4423-9c6f-09acbfabf5b6"
    dest_transfer_id = "2fde89c0-6fb4-11eb-8c47-0eb1aa8d4337"

    source = caws.Endpoint(source_name,
                           source_compute_id,
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

    transfer_manager = TransferManager([source, destination], log_level=logging.DEBUG)
    transfer_manager.start()
    assert transfer_manager.started

    files_by_src = {source_name: [source_path,]}
    transfer_record = transfer_manager.transfer(files_by_src,
                              destination,
                              "test_transfer",
                              True,
                              success_callback,
                              failure_callback)

    transfer_manager.shutdown()
    assert not transfer_manager.started
    assert transfer_record.status == TransferStatus.COMPLETED


    
