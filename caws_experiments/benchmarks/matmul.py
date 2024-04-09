import numpy as np
from tempfile import NamedTemporaryFile

from caws.task import caws_task
from caws.path import CawsPath
from caws.features import ArgAttrFeature

def matrix_multiplication(a_path, b_path, *, _caws_output_dir):
    import numpy as np
    a = np.load(a_path)
    b = np.load(b_path)

    c = a @ b

    output_path = os.path.join(_caws_output_dir, "output.npy")
    with open(output_path, 'wb') as fp:
        np.save(fp, c)
        
    return output_path
func = caws_task(matrix_multiplication, features=[ArgAttrFeature("size")])

def generate_inputs(src_endpoint, size, data_dir=""):
    size_generators = {
        'test' : 64,
        'small' : 512,
        'large': 4096,
        '1': 64,
        '2': 128,
        '3': 256,
        '4': 384,
        '5': 512
    }
    dim = size_generators[size]
    a = np.random.rand(dim, dim)
    mat_a_file = "mat_a.npy"
    np.save(mat_a_file, a)
    mat_a_path = CawsPath(src_endpoint, mat_a_file, isolate=True)

    b = np.random.rand(dim, dim)
    mat_b_file = "mat_b.npy"
    np.save(mat_b_file, b)
    mat_b_path = CawsPath(src_endpoint, mat_b_file, isolate=True)
    return (mat_a_path, mat_b_path), {}