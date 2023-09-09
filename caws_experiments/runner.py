import json
import tempfile
import concurrent.futures
import time

import click

from caws_experiments.benchmarks import utils as benchmark_utils
from caws_experiments import utils

import caws
from caws.strategy import FCFS_RoundRobin
from caws.predictors.transfer_predictors import TransferPredictor

@click.group()
def cli():
    pass

@cli.command()
@click.argument("benchmark_name", type=str)
@click.option(
    "endpoint_name", "-e,"
    required=True,
    type=str, 
    help="Name of endpoint in config file"
)
@click.option(
    "--benchmark-input-size", "-s",
    required=True,
    type=click.Choice(["test", "small", "large"])
    help="Size of the benchmark to run (test, small, or large)"
)
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
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
    '--warmup', '-w',
    is_flag=True,
    help="Warmup the executor before running tasks"
)
def benchmark(benchmark_name,
              benchmark_input_size,
              endpoint_name,
              config,
              data_dir,
              ntasks,
              warmup):
    config_obj = json.load(open(config, "r"))
    benchmark = benchmark_utils.import_benchmark(benchmark_name)
    src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])
    args, kwargs = benchmark.generate_inputs(src_endpoint, benchmark_input_size, data_dir=data_dir)
    func = benchmark_utils.mainify(benchmark.func)

    endpoint = utils.load_endpoint(config_obj, endpoint_name)
    endpoints = [endpoint,]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy, caws_database_url=config_obj["caws_monitoring_db"]) as executor:
        if warmup:
            fut = executor.submit(time.sleep, 5)
            fut.result()
        
        futures = []
        for _ in range(ntasks):
            futures.append(executor.submit(func, *args, **kwargs))
        concurrent.futures.wait(futures)

        for future in futures:
            future.result()

@cli.command()
@click.argument("endpoint_name", type=str)
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--benchmark-input-size", "-s",
    required=True,
    type=click.Choice(["test", "small", "large"])
    help="Size of the benchmark to run (test, small, or large)"
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
    help="Number of each task to run",
)
@click.option(
    "--warmup", "-w", 
    is_flag=True, 
    help="Warmup the executor before running tasks"
)
@click.option(
    "--include", "-i",
    multiple=True,
    help="Benchmarks to include."
)
@click.option(
    "--exclude", "-e",
    multiple=True,
    help="Benchmarks to exclude."
)
def profile(endpoint_name, 
            config,
            benchmark_input_size,
            data_dir,
            ntasks,
            warmup,
            include,
            exclude):
    config_obj = json.load(open(config, "r"))
    if len(include) > 0:
        benchmark_names = include
    else:
        benchmark_names = ["bfs", "compression", "dna", "inference", "mst", "pagerank", "thumbnail", "video"]
    
    src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])
    benchmarks = []
    for benchmark_name in benchmarks:
        benchmark = benchmark_utils.import_benchmark(benchmark_name)
        args, kwargs = benchmark.generate_inputs(src_endpoint, benchmark_input_size, data_dir=data_dir)
        func = benchmark_utils.mainify(benchmark.func)
        benchmarks.append((func, args, kwargs))

    endpoint = utils.load_endpoint(config_obj, endpoint_name)
    endpoints = [endpoint,]
    strategy = FCFS_RoundRobin(endpoints, TransferPredictor(endpoints))
    with caws.CawsExecutor(endpoints, strategy, caws_database_url=config_obj["caws_monitoring_db"]) as executor:
        if warmup:
            fut = executor.submit(time.sleep, 5)
            fut.result()
        
        futures = []
        for func, args, kwargs in benchmarks:
            for _ in range(ntasks):
                futures.append(executor.submit(func, *args, **kwargs))
        concurrent.futures.wait(futures)

        for future in futures:
            future.result()

if __name__ == "__main__":
    cli()