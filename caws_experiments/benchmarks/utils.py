import importlib

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

def import_benchmark(name:str):
    return importlib.import_module(f".{name}", "caws_experiments.benchmarks")