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