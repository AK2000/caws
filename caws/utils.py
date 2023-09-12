from functools import wraps

from globus_compute_sdk import Client

client = Client()

def create_output_dir_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import os
        os.makedirs(kwargs["_caws_output_dir"])
        return func(*args, **kwargs)
    return wrapper

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