import os

from caws.path import CawsPath
'''
    Generate test, small and large workload for compression test.

    
    :param size: workload size
    :param src_endpoint
    :param data_dir: directory where benchmark data is placed
'''
def generate_inputs(src_endpoint, size, data_dir=""):
    data = os.path.join(data_dir, "300.utilities", "311.compression")
    data_path = CawsPath(src_endpoint, data, isolate=True)

    return (data_path,), {}