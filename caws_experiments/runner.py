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

@click.command()
@click.argument("benchmark_name", type=str)
@click.argument(
    "benchmark-input-size", type=click.Choice(["test", "small", "large"])
)
@click.argument(
    "endpoint_name", type=str, #help="Name of endpoint in config file"
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
@click.option('--warmup', '-w', is_flag=True, help="Warmup the executor before running tasks")
def run_benchmark(benchmark_name,
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

if __name__ == "__main__":
    run_benchmark()