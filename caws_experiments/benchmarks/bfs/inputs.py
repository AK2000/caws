size_generators = {
    'test' : 10,
    'small' : 10000,
    'large': 100000
}

def generate_inputs(src_endpoint, size):
    return (size_generators[size], ), {}