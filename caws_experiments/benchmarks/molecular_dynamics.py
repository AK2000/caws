from caws.task import caws_task
from chemfunctions import compute_vertical

func = caws_task(compute_vertical)

def generate_inputs(src_endpoint, size, data_dir=""):
    size_generators = {
        'test' : "CCC",
        'small' : "C#CC12OC3CC1C2O3",
        'large': "CC1=NC(O)=NC=C1F",
    }
    return (size_generators[size], ), {}