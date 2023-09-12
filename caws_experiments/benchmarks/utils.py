import importlib

def import_benchmark(name:str):
    return importlib.import_module(f".{name}", "caws_experiments.benchmarks")