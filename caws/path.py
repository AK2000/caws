import caws

import os

class CawsPath:
    endpoint: caws.Endpoint
    source_path: str

    def __init__(self, src_endpoint: caws.Endpoint, src_path: str):
        self.endpoint = src_endpoint
        self.source_path = src_path

    def get_src_path(self):
        return os.path.join(self.endpoint.local_path, self.source_path)

    def get_dest_path(self, dst_endpoint):
        return os.path.join(dst_endpoint.local_path, self.source_path)

def to_caws_path(src_endpoint: caws.Endpoint, host_path: str):
    pass