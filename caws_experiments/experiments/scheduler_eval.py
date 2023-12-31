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
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.strategy.mhra import MHRA
from caws.strategy.cluster_mhra import ClusterMHRA
from caws.predictors.predictor import Predictor
from caws.task import CawsTask, CawsTaskInfo
from caws.path import CawsPath

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

def shadow_run(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url):
    tasks = []
    for func, args, kwargs in benchmarks:
        for _ in range(ntasks):
            tasks.append(create_task_info(func, *args, **kwargs))
    
    print(f"Starting timing strategies with {len(tasks)} tasks")
    predictor.start()
    start = time.time()
    schedule, _ = strategy.schedule(tasks)
    total_time = time.time() - start

    transfer_time, transfer_energy = strategy.calculate_transfer(schedule)

    return {
        "scheduler_overhead": total_time,
        "pred_transfer_energy": transfer_energy
    }

def run_one(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url):

    with caws.CawsExecutor(endpoints, 
                           strategy,
                           caws_database_url=monitoring_url,
                           predictor=predictor) as executor:
        

        start_time = datetime.datetime.now()
        futures = []
        with executor.scheduling_lock:
            for i, (func, args, kwargs) in tqdm(enumerate(benchmarks)):
                for _ in range(ntasks):
                    futures.append(executor.submit(func, *args, **kwargs))
        
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

        end_time = datetime.datetime.now()

        for future in futures:
            future.result() # Raise any exceptions

    # Collect energy use for each endpoint
    total_energy = 0
    for endpoint in endpoints:
        _, _, energy = endpoint.collect_monitoring_info(start_time)
        total_energy += energy["total_energy"].dropna().sum()

    runtime = (end_time - start_time).total_seconds()

    return {"runtime": runtime, "energy": total_energy}


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
    "--ntasks", "-n",
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
    "--include", "-i",
    multiple=True,
    help="Benchmarks to include."
)
@click.option(
    "--mock", "-m", 
    is_flag=True, 
    help="Run the scheduler without executing tasks"
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="scheduler_eval.jsonl",
    help="Place to store results file"
)
def compare(config,
        endpoints,
        data_dir,
        ntasks,
        exclude,
        include,
        mock,
        result_path):

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
    
    if len(include) > 0:
        benchmark_names = include
    else:
        benchmark_names = ["bfs", "compression", "dna", "inference", "mst", "pagerank", "thumbnail", "video"]
    benchmark_names = [b for b in benchmark_names if b not in exclude]
    benchmarks = []
    for benchmark_name in benchmark_names:
        print("Setting Up Benchmark:", benchmark_name)
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        args, kwargs = benchmark.generate_inputs(src_endpoint, "small", data_dir=data_dir)
        benchmark.func.mainify()
        benchmarks.append((benchmark.func, args, kwargs))
        
    monitoring_url = config_obj["caws_monitoring_db"]
    predictor = Predictor(endpoints, monitoring_url)

    if mock:
        run_one = shadow_run
    
    print(f"Testing MHRA Strategy")
    strategy = MHRA(endpoints, predictor, alpha=0.5)
    results = run_one(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url)
    results["strategy"] = "mhra"
    results["alpha"] = 0.5
    results["ntasks"] = ntasks
    with open(result_path, "a") as fp:
        fp.write(json.dumps(results))
        fp.write("\n")

    print("Rate limiting so endpoints are cold")
    time.sleep(60)


    print(f"Testing ClusterMHRA Strategy")
    strategy = ClusterMHRA(endpoints, predictor, alpha=0.5)
    results = run_one(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url)
    results["strategy"] = "cluster_mhra"
    results["alpha"] = 0.5
    results["ntasks"] = ntasks
    with open(result_path, "a") as fp:
        fp.write(json.dumps(results))
        fp.write("\n")

    print("Rate limiting so endpoints are cold")
    time.sleep(60)

    print(f"Running tasks in a round robin fashion")
    strategy = FCFS_RoundRobin(endpoints, predictor)
    results = run_one(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url)
    results["strategy"] = "round_robin"
    results["alpha"] = None
    results["ntasks"] = ntasks
    with open(result_path, "a") as fp:
        fp.write(json.dumps(results))
        fp.write("\n")
    
    print("Rate limiting so endpoints are cold")
    time.sleep(60)
    
    for endpoint in endpoints:
        print(f"Submitting all tasks to endpoint {endpoint.name}")
        strategy = FCFS_RoundRobin([endpoint,], predictor)
        results = run_one([endpoint,], strategy, predictor, benchmarks, ntasks, monitoring_url)
        results["strategy"] = endpoint.name
        results["alpha"] = None
        results["ntasks"] = ntasks

        with open(result_path, "a") as fp:
            fp.write(json.dumps(results))
            fp.write("\n")    

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
    "--ntasks", "-n",
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
    "--include", "-i",
    multiple=True,
    help="Benchmarks to include."
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="scheduler_eval.jsonl",
    help="Place to store results file"
)
def sensitivity(config,
                endpoints,
                data_dir,
                ntasks,
                exclude,
                include,
                result_path):

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
    
    if len(include) > 0:
        benchmark_names = include
    else:
        benchmark_names = ["bfs", "compression", "dna", "inference", "mst", "pagerank", "thumbnail", "video"]
    benchmark_names = [b for b in benchmark_names if b not in exclude]
    benchmarks = []
    for benchmark_name in benchmark_names:
        print("Setting Up Benchmark:", benchmark_name)
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        args, kwargs = benchmark.generate_inputs(src_endpoint, "small", data_dir=data_dir)
        benchmark.func.mainify()
        benchmarks.append((benchmark.func, args, kwargs))
        
    monitoring_url = config_obj["caws_monitoring_db"]
    predictor = Predictor(endpoints, monitoring_url)

    alphas = [1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0]

    for alpha in alphas:
        strategy = ClusterMHRA(endpoints, predictor, alpha=alpha)
        results = run_one(endpoints, strategy, predictor, benchmarks, ntasks, monitoring_url)
        results["strategy"] = "cluster_mhra"
        results["alpha"] = alpha
        results["ntasks"] = ntasks
        with open(result_path, "a") as fp:
            fp.write(json.dumps(results))
            fp.write("\n")

        print("Rate limiting to ensure endpoints are cold.")
        time.sleep(60)

if __name__ == "__main__":
    cli()
