import numpy as np
from tempfile import NamedTemporaryFile

from caws.task import caws_task
from caws.path import CawsPath
from caws.features import ArgAttrFeature

def cholesky_decomposition(a_path, *, _caws_output_dir):
    import numpy as np
    from numpy.linalg import cholesky

    a = np.load(a_path)
    L = cholesky(a)
    output_path = os.path.join(_caws_output_dir, "output.npy")
    with open(output_path, 'wb') as fp:
        np.save(fp, L)
        
    return output_path

func = caws_task(cholesky_decomposition, features=[ArgAttrFeature("size")])

def generate_inputs(src_endpoint, size, data_dir=""):
    size_generators = {
        'test' : 64,
        'small' : 512,
        'large': 4096,
    }
    dim = size_generators[size]
    a = np.random.rand(dim, dim)
    a = a.T @ a # Make hermatian
    mat_a_file = "mat_cholesky.npy"
    np.save(mat_a_file, a)
    mat_a_path = CawsPath(src_endpoint, mat_a_file, isolate=True)
    return (mat_a_path,), {}