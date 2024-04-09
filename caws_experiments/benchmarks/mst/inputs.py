size_generators = {
    'test' : 10,
    'small' : 10000,
    'large': 500000,
    '1': 10000,
    '2': 25000,
    '3': 50000,
    '4': 75000,
    '5': 100000
}

def generate_inputs(src_endpoint, size, data_dir=""):
    return (size_generators[size], ), {}