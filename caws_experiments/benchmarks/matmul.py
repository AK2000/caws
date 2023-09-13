import numpy as np

from caws.task import caws_task
from caws.features import ArgAttrFeature

def matrix_multiplication(a, b):
    return a @ b
func = caws_task(matrix_multiplication, features=[ArgAttrFeature("size")])

def generate_inputs(src_endpoint, size, data_dir=""):
    size_generators = {
        'test' : 64,
        'small' : 512,
        'large': 2048
    }
    dim = size_generators[size]
    a = np.random.rand(dim, dim)
    b = np.random.rand(dim, dim)
    return (a, b), {}