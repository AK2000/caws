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

def loop(iters: int):
    s = 0
    for i in range(iters):
        s += i
    return s
loop = mainify(loop)

def transfer_file(path, _caws_output_dir):
    import os

    # First try opening a file transferred in
    with open(path) as f:
        pass

    # Now try creating a file output
    os.makedirs(_caws_output_dir, exist_ok=True)
    output_path = os.path.join(_caws_output_dir, "output.txt")
    with open(output_path, "w") as f:
        f.write("Hello World!\n")

    return output_path

transfer_file = mainify(transfer_file)

