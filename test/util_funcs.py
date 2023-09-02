def mainify(obj):
    """If obj is not defined in __main__ then redefine it in 
    main so that dill will serialize the definition along with the object"""
    if obj.__module__ != "__main__":
        import __main__
        import inspect
        s = inspect.getsource(obj)
        co = compile(s, '<string>', 'exec')
        exec(co, __main__.__dict__)
        return __main__.__dict__.get(obj.__name__)
    return obj

# Define a small set of functions for testing

def add(a: int, b: int):
    return a + b
add = mainify(add) # Fix serialization scope so the function does not have to be installed on the endpoint

def sleep(sec: float):
    import time
    time.sleep(sec)
    return
sleep = mainify(sleep)

def gemm(dim: int):
    import numpy as np

    A = np.random.rand(dim, dim)
    B = np.random.rand(dim, dim)

    return A @ B
gemm = mainify(gemm)

def transfer_file(_globus_files):
    return True
transfer_file = mainify(transfer_file)

