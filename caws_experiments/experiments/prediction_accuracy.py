import json
import tempfile
import concurrent.futures
import time
from tqdm import tqdm
from collections import defaultdict

import sqlalchemy
from sqlalchemy import text, bindparam
import pandas as pd
import click
import numpy as np

import caws
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.predictors.predictor import Predictor

from caws_experiments.benchmarks import utils as benchmark_utils
from caws_experiments import utils

def create_data_history(endpoints, src_endpoint, data_dir, benchmark_names, sizes, monitoring_url, ntasks):
    print("Colecting Data")

    benchmarks = []
    for benchmark_name in benchmark_names:
        print("Setting Up Benchmark:", benchmark_name)
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        benchmark.func.mainify()
        for size in sizes:
            args, kwargs = benchmark.generate_inputs(src_endpoint, size, data_dir=data_dir)
            benchmarks.append((benchmark.func, args, kwargs))

    predictor = Predictor(endpoints, monitoring_url)
    strategy = FCFS_RoundRobin(endpoints, predictor)
    with caws.CawsExecutor(endpoints, 
                           strategy,
                           caws_database_url=monitoring_url) as executor:

        print("Warming up endpoint!")
        fut = executor.submit(time.sleep, 5)
        fut.result()

        futures = []
        for i, (func, args, kwargs) in tqdm(enumerate(benchmarks)):
            # Ensure all tasks are batched together
            with executor.scheduling_lock:
                for _ in range(ntasks):
                    futures.append(executor.submit(func, *args, **kwargs))
            
        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

    print("Completed!")

def measure_accuracy(predictor, endpoints, src_endpoint, data_dir, benchmark_names, input_size, monitoring_url, ntasks):
    print("Measuring accuracy")

    benchmarks = []
    for benchmark_name in benchmark_names: 
        print("Setting Up Benchmark:", benchmark_name)
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        benchmark.func.mainify()
        args, kwargs = benchmark.generate_inputs(src_endpoint, input_size, data_dir=data_dir)
        benchmarks.append((benchmark.func, args, kwargs))

    predictor = Predictor(endpoints, monitoring_url)
    strategy = FCFS_RoundRobin(endpoints, predictor)

    predictions_by_benchmark = dict()
    task_ids = []
    with caws.CawsExecutor(endpoints, 
                           strategy,
                           caws_database_url=monitoring_url,
                           predictor=predictor) as executor:

        print("Warming up endpoint!")
        fut = executor.submit(time.sleep, 5)
        fut.result()

        for i, (func, args, kwargs) in tqdm(enumerate(benchmarks)):
            futures = []
            with executor.scheduling_lock:
                for _ in range(ntasks):
                    futures.append(executor.submit(func, *args, **kwargs))

            predictions_by_benchmark[futures[0].task_info.function_name] = predictor.predict_execution(endpoints[0], futures[0].task_info)

            concurrent.futures.wait(futures)
            for future in futures:
                future.result() # Raise any exceptions
                task_ids.append(future.task_info.task_id)

    engine = sqlalchemy.create_engine(monitoring_url)
    with engine.begin() as connection:
        query = text("""SELECT func_name, running_duration, energy_consumed """
                """FROM caws_task """
                """WHERE (caws_task_id in :task_ids)""")
        query = query.bindparams(bindparam("task_ids", task_ids, expanding=True))
        task_measurements = pd.read_sql(query, connection).dropna()

    task_measurements["size"] = input_size
    task_measurements = task_measurements.merge(pd.Series({k: v.runtime for k,v in predictions_by_benchmark.items()}, name="runtime_pred"), left_on="func_name", right_index=True)
    task_measurements = task_measurements.merge(pd.Series({k: v.energy for k,v in predictions_by_benchmark.items()}, name="energy_pred"), left_on="func_name", right_index=True)
    task_measurements["runtime_error"] = (task_measurements["running_duration"] - task_measurements["runtime_pred"]).abs()
    task_measurements["energy_error"] = (task_measurements["energy_consumed"] - task_measurements["energy_pred"]).abs()

    return task_measurements


@click.group()
def cli():
    pass

@cli.command(name="features")
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--endpoint_name", "-l",
    required=False,
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
    help="Location of experiment config.",
)
@click.option(
    "--exclude", "-e",
    multiple=True,
    help="Benchmarks to exclude."
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="results.csv",
    help="Place to store results file"
)
def feature_prediction_accuracy(config,
                                endpoint_name,
                                data_dir,
                                ntasks,
                                exclude,
                                result_path):

    config_obj = json.load(open(config, "r"))
    benchmark_names = ["bfs", "mst", "pagerank", "matmul"]
    benchmark_names = [b for b in benchmark_names if b not in exclude]
    train_sizes = ["1", "3", "4"]
    pred_sizes = ["2", "5"]

    endpoints = []
    endpoint = utils.load_endpoint(config_obj, endpoint_name)
    endpoints.append(endpoint)

    if endpoint_name == config_obj["host"]:
        src_endpoint = endpoint
    else:
        src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])
    
    
    monitoring_url = config_obj["caws_monitoring_db"]
    predictor = Predictor(endpoints, monitoring_url)
    # Hacky way of removing historical data to see what prediction would look like
    predictor.start()
    for endpoint_model in predictor.endpoint_models.values():
        endpoint_model.tasks.drop(endpoint_model.tasks.index, inplace=True)

    create_data_history(endpoints, src_endpoint, data_dir, benchmark_names, train_sizes, monitoring_url, ntasks)

    print("Updating predictor")
    predictor.update()
    
    measurements = []
    for endpoint in endpoints:
        for size in pred_sizes:
            execution_times = measure_accuracy(predictor, [endpoint,], src_endpoint, data_dir, benchmark_names, size, monitoring_url, ntasks)
            execution_times["endpoint"] = endpoint.name
            measurements.append(execution_times)

    results = pd.concat(measurements)
    results.to_csv(result_path)

if __name__ == "__main__":
    cli()
