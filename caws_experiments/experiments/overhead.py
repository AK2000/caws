import json
import tempfile
import concurrent.futures
import time
from tqdm import tqdm
from collections import defaultdict
import datetime
import uuid

import sqlalchemy
from sqlalchemy import text, bindparam
import pandas as pd
import click
import numpy as np

import caws
from caws.task import CawsTask, CawsTaskInfo
from caws.path import CawsPath
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.strategy.mhra import MHRA
from caws.strategy.cluster_mhra import ClusterMHRA
from caws.predictors.transfer_predictors import TransferPredictor
from caws.predictors.predictor import Predictor

from caws_experiments.benchmarks import utils as benchmark_utils
from caws_experiments import utils


def create_task_info(fn, *args, ** kwargs):
    if isinstance(fn, CawsTask):
        features = fn.extract_features(*args, **kwargs)
        fn = fn.extract_func()
    else:
        features = []

    task_id = str(uuid.uuid4())
    name = kwargs.get("name", f"{fn.__module__}.{fn.__qualname__}")
    task_info = CawsTaskInfo(fn, list(args), kwargs, task_id, name, features=features)
    for i, arg in enumerate(task_info.task_args):
        if isinstance(arg, CawsPath):
            task_info.transfer_size[arg.endpoint.transfer_endpoint_id] += arg.size
            task_info.transfer_files[arg.endpoint.transfer_endpoint_id] += arg.num_files

    for key, arg in task_info.task_kwargs.items():
        if isinstance(arg, CawsPath):
            task_info.transfer_size[arg.endpoint.transfer_endpoint_id] += arg.size
            task_info.transfer_files[arg.endpoint.transfer_endpoint_id] += arg.num_files
    
    return task_info

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--endpoints", "-l",
    required=False,
    multiple=True,
    help="Endpoint to use. Default (all)"
)
@click.option(
    "--data_dir", "-d",
    type=str,
    default="serverless-benchmarks-data",
    help="Location of data directory for benchmarks",
)
@click.option(
    "--max_tasks", "-n",
    type=int,
    default=1,
    help="Number of each benchmark to include in the mix",
)
@click.option(
    "--exclude", "-e",
    multiple=True,
    help="Benchmarks to exclude."
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="scheduler_overhead.jsonl",
    help="Place to store results file"
)
def scheduler_overhead(config, endpoints, data_dir, max_tasks, exclude, result_path):
    config_obj = json.load(open(config, "r"))

    endpoint_names = endpoints if len(endpoints) > 0 else config_obj["endpoints"].keys()
    endpoints = []
    src_endpoint = None
    for endpoint_name in endpoint_names:
        endpoint = utils.load_endpoint(config_obj, endpoint_name)
        endpoints.append(endpoint)
        if endpoint_name == config_obj["host"]:
            src_endpoint = endpoint

    if src_endpoint is None:
        src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])

    monitoring_url = config_obj["caws_monitoring_db"]
    predictor = Predictor(endpoints, monitoring_url)
    tp = TransferPredictor(endpoints)
    startegies = {
        "mhra": MHRA(endpoints, predictor, alpha=0.5), 
        "cluster_mhra": ClusterMHRA(endpoints, predictor, alpha=0.5),
        "round_robin": FCFS_RoundRobin(endpoints, tp)
    }

    benchmark_names = ["bfs", "compression", "dna", "inference", "mst", "pagerank", "thumbnail", "video", "matmul"]
    benchmark_names = [b for b in benchmark_names if b not in exclude]
    benchmarks = []
    for benchmark_name in benchmark_names:
        print("Setting Up Benchmark:", benchmark_name)
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        args, kwargs = benchmark.generate_inputs(src_endpoint, "small", data_dir=data_dir)
        benchmark.func.mainify()
        benchmarks.append((benchmark.func, args, kwargs))

    predictor.start()
    task_range = np.logspace(0, np.log2(max_tasks), num=int(np.log2(max_tasks))+1, base=2, dtype='int', endpoint=True)
    for ntasks in task_range:
        tasks = []
        for func, args, kwargs in benchmarks:
            for _ in range(ntasks):
                tasks.append(create_task_info(func, *args, **kwargs))
        
        print(f"Starting timing strategies with {len(tasks)} tasks")

        for name, strategy in tqdm(startegies.items()):
            start = time.time()
            strategy.schedule(tasks)
            runtime = time.time() - start

            with open(result_path, "a") as fp:
                fp.write(json.dumps({"strategy": name, "ntasks": len(tasks), "runtime": runtime}))
                fp.write("\n")

        print("Completed one round")
    print("Completed test!")
    


if __name__ == "__main__":
    cli()
