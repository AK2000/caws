import caws

import os
import uuid

class CawsPath:
    endpoint: caws.Endpoint
    source_path: str
    size: int

    def __init__(self, src_endpoint: caws.Endpoint, src_path: str, replicate_path: bool = False, isolate: bool = False):
        rel_path = os.path.relpath(src_path, src_endpoint.local_path)
        self.endpoint = src_endpoint
        self.source_path = rel_path
        self.replicate_path = replicate_path
        self.isolate = isolate
        self.size = self._get_size(self.get_src_local_path())

    def _get_size(self, path):
        if os.path.isfile(path):
            return os.path.getsize(path)

        size = 0
        for root, dirs, files in os.walk(path):
            size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
        return size

    def get_src_endpoint_path(self):
        return os.path.join(self.endpoint.endpoint_path, self.source_path)

    def get_src_local_path(self):
        return os.path.join(self.endpoint.local_path, self.source_path)

    def _get_endpoint_path(self, base_path, task_id):
        path = os.path.join(base_path, ".caws")
        if self.isolate:
            if task_id is None:
                raise Exception("Cannot create isolated file without task id")
            path = os.path.join(path, task_id)
        
        if self.replicate_path:
            path = os.path.join(path, self.source_path)
        else:
            path = os.path.join(path, os.path.basename(self.source_path))

        return path

    def get_dest_endpoint_path(self, dst_endpoint, task_id = None):
        return self._get_endpoint_path(dst_endpoint.endpoint_path, task_id)

    def get_dest_local_path(self, dst_endpoint, task_id = None):
        return self._get_endpoint_path(dst_endpoint.local_path, task_id)