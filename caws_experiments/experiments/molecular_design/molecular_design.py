import json
import tempfile
import concurrent.futures
from concurrent.futures import as_completed
import time
import datetime
from tqdm import tqdm

import click
import numpy as np
import pandas as pd

import caws
from caws.strategy.round_robin import FCFS_RoundRobin
from caws.strategy.mhra import MHRA
from caws.strategy.cluster_mhra import ClusterMHRA
from caws.predictors.predictor import Predictor
from caws.task import caws_task
from caws.features import ArgLenFeature

from caws_experiments import utils
from chemfunctions import compute_vertical

from globus_compute_sdk import Executor

def train_model(train_data):
    """Train a machine learning model using Morgan Fingerprints.
    
    Args:
        train_data: Dataframe with a 'smiles' and 'ie' column
            that contains molecule structure and property, respectfully.
    Returns:
        A trained model
    """
    # Imports for python functions run remotely must be defined inside the function
    from chemfunctions import MorganFingerprintTransformer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    
    
    model = Pipeline([
        ('fingerprint', MorganFingerprintTransformer()),
        ('knn', KNeighborsRegressor(n_neighbors=4, weights='distance', metric='jaccard', n_jobs=-1))  # n_jobs = -1 lets the model run all available processors
    ])
    
    return model.fit(train_data['smiles'], train_data['ie'])

train_model = caws_task(train_model, features=[ArgLenFeature(0)])

def run_model(model, smiles):
    """Run a model on a list of smiles strings
    
    Args:
        model: Trained model that takes SMILES strings as inputs
        smiles: List of molecules to evaluate
    Returns:
        A dataframe with the molecules and their predicted outputs
    """
    import pandas as pd
    pred_y = model.predict(smiles)
    return pd.DataFrame({'smiles': smiles, 'ie': pred_y})

def combine_inferences(inputs=[]):
    """Concatenate a series of inferences into a single DataFrame
    Args:
        inputs: a list of the component DataFrames
    Returns:
        A single DataFrame containing the same inferences
    """
    import pandas as pd
    df = pd.concat(inputs, ignore_index=True)
    df.sort_values('ie', ascending=False, inplace=True)
    return df

@click.group()
def cli():
    pass

@cli.command()
@click.argument("endpoint_name", type=str)
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--search_space", "-s",
    type=str,
    default="experiments/molecular_design/QM9-search.tsv",
    help="File containing total search space",
)
@click.option(
    "--num_tasks", "-n",
    type=int,
    default=1,
    help="Number of each task to run",
)
@click.option(
    "--num_examples", "-e",
    type=int,
    default=128,
    help="Max number of examples to profile training with",
)
@click.option(
    "--warmup", "-w", 
    is_flag=True, 
    help="Warmup the executor before running tasks"
)
def profile(endpoint_name, 
            config,
            search_space,
            num_tasks,
            num_examples,
            warmup):

    config_obj = json.load(open(config, "r"))
    src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])
    search_space = pd.read_csv(search_space, delim_whitespace=True)

    endpoint = utils.load_endpoint(config_obj, endpoint_name)
    endpoints = [endpoint,]
    predictor = Predictor(endpoints, config_obj["caws_monitoring_db"])
    strategy = FCFS_RoundRobin(endpoints, predictor)

    with caws.CawsExecutor(endpoints, 
                           strategy,
                           caws_database_url=config_obj["caws_monitoring_db"],
                           predictor=predictor) as executor:
        if warmup:
            # Make sure we are not getting a cold endpoint
            fut = executor.submit(time.sleep, 5)
            fut.result()
        
        # Profile compute vertical
        smiles = search_space.sample(num_tasks)['smiles']
        with executor.scheduling_lock:
            futures = [executor.submit(compute_vertical, s) for s in smiles]
        concurrent.futures.wait(futures)

        # Create training data
        train_data = []
        for future in futures:
            # Get the input
            smiles = future.task_info.task_args[0]
            
            # Check if the run completed successfully
            if future.exception() is not None:
                pass
            else:
                # If it succeeded, store the result
                print(f'Computation for {smiles} succeeded')
                train_data.append({
                    'smiles': smiles,
                    'ie': future.result(),
                    'batch': 0,
                    'time': time.monotonic()
                })
        train_data = pd.DataFrame(train_data)

        # Profile train synchronously
        example_range = np.logspace(2, np.log2(num_examples), num=int(np.log2(num_examples))-1, base=2, dtype='int', endpoint=True)

        for n_examples in example_range:
            examples = train_data.sample(n_examples, replace=True) # Generate examples
            #model = train_model(examples)
            future = executor.submit(train_model, examples)
            model = future.result()

        # Profile inference
        chunks = np.array_split(search_space['smiles'], 64)
        inference_futures = []
        with executor.scheduling_lock:
            for _ in range(0, num_tasks, len(chunks)):
                inference_futures.extend([executor.submit(run_model, model, chunk) for chunk in chunks])
        concurrent.futures.wait(inference_futures)

        # Profile combine inference
        inputs = [f.result() for f in inference_futures[:len(chunks)]]
        with executor.scheduling_lock:
            combine_futures = [executor.submit(combine_inferences, inputs) for _ in range(num_tasks)]
        concurrent.futures.wait(combine_futures)


@cli.command()
@click.argument("strategy_name", type=str)
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--search_space", "-s",
    type=str,
    default="experiments/molecular_design/QM9-search.tsv",
    help="File containing total search space",
)
@click.option(
    "--initial", "-i",
    type=int,
    default=256,
    help="Initial dataset size",
)
@click.option(
    "--batch_size", "-b",
    type=int,
    default=128,
    help="Number of simulations to run each round",
)
@click.option(
    "--total", "-t",
    type=int,
    default=1024,
    help="Total molecules to evaluate",
)
@click.option(
    "--endpoints", "-l",
    required=False,
    multiple=True,
    help="Endpoint to use. Default (all)"
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="molecular_design_eval.jsonl",
    help="Place to store results file"
)
def run(strategy_name, 
        config,
        search_space,
        initial,
        batch_size,
        total,
        endpoints,
        result_path):

    config_obj = json.load(open(config, "r"))
    # src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])
    search_space = pd.read_csv(search_space, delim_whitespace=True)

    if strategy_name in config_obj["endpoints"]:
        endpoint = utils.load_endpoint(config_obj, strategy_name)
        endpoints = [endpoint,]
        predictor = Predictor(endpoints, config_obj["caws_monitoring_db"])
        strategy = FCFS_RoundRobin(endpoints, predictor)
    else:
        endpoint_names = endpoints if len(endpoints) > 0 else config_obj["endpoints"].keys()
        endpoints = []
        for endpoint_name in endpoint_names:
            endpoint = utils.load_endpoint(config_obj, endpoint_name)
            endpoints.append(endpoint)
            # if endpoint_name == config_obj["host"]:
            #     src_endpoint = endpoint

        predictor = Predictor(endpoints, config_obj["caws_monitoring_db"])
        if strategy_name == "round_robin":
            strategy = FCFS_RoundRobin(endpoints, predictor)
        elif strategy_name == "mhra":
            strategy = MHRA(endpoints, predictor)
        elif strategy_name == "cluster_mhra":
            strategy = ClusterMHRA(endpoints, predictor, alpha=0.2)

    # if src_endpoint == None:
    #     src_endpoint = utils.load_endpoint(config_obj, config_obj["host"])

    with caws.CawsExecutor(endpoints, 
                           strategy,
                           caws_database_url=config_obj["caws_monitoring_db"],
                           predictor=predictor) as executor:

        with tqdm(total=total) as prog_bar:
            start_time = datetime.datetime.now()
            
            already_ran = set()
            train_data = []

            # Profile compute vertical
            init_mols = search_space.sample(initial)['smiles']
            with executor.scheduling_lock:
                sim_futures = [executor.submit(compute_vertical, s) for s in init_mols]
            
            # Loop until you finish populating the initial set
            while len(sim_futures) > 0: 
                # First, get the next completed computation from the list
                future = next(as_completed(sim_futures))

                # Remove it from the list of still-running tasks
                sim_futures.remove(future)

                # Get the input 
                smiles = future.task_info.task_args[0]
                already_ran.add(smiles)

                # Check if the run completed successfully
                if future.exception() is not None:
                    # If it failed, pick a new SMILES string at random and submit it    
                    smiles = search_space.sample(1).iloc[0]['smiles'] # pick one molecule
                    new_future = executor.submit(compute_vertical, smiles) # launch a simulation in Parsl
                    sim_futures.append(new_future) # store the Future so we can keep track of it
                else:
                    # If it succeeded, store the result
                    prog_bar.update(1)
                    train_data.append({
                        'smiles': smiles,
                        'ie': future.result(),
                        'batch': 0,
                        'time': (datetime.datetime.now() - start_time).total_seconds()
                    })

            train_data = pd.DataFrame(train_data)

            # Loop until complete
            batch = 1
            while len(train_data) < total:
                # Train and predict as show in the previous section.
                train_future = executor.submit(train_model, train_data)
                model = train_future.result()
                inference_futures = [executor.submit(run_model, model, chunk) for chunk in np.array_split(search_space['smiles'], 64)]
                predictions = executor.submit(combine_inferences, inputs=[f.result() for f in inference_futures]).result()
                
                sim_futures = []
                for smiles in predictions['smiles']:
                    if smiles not in already_ran:
                        sim_futures.append(executor.submit(compute_vertical, smiles))
                        already_ran.add(smiles)
                        if len(sim_futures) >= batch_size:
                            break

                # Wait for every task in the current batch to complete, and store successful results
                new_results = []
                for future in as_completed(sim_futures):
                    if future.exception() is None:
                        prog_bar.update(1)
                        new_results.append({
                            'smiles': future.task_info.task_args[0],
                            'ie': future.result(),
                            'batch': batch, 
                            'time': (datetime.datetime.now() - start_time).total_seconds()
                        })
                        
                # Update the training data and repeat
                batch += 1
                train_data = pd.concat((train_data, pd.DataFrame(new_results)), ignore_index=True)
            
            end_time = datetime.datetime.now()

    runtime = (end_time - start_time).total_seconds()
    total_energy = 0
    for endpoint in endpoints:
        _, _, energy = endpoint.collect_monitoring_info(start_time)
        total_energy += energy["total_energy"].dropna().sum()

    results = {"strategy": strategy_name, "runtime": runtime, "total_energy": total_energy}
    with open(result_path, "a") as fp:
        fp.write(json.dumps(results))
        fp.write("\n")

if __name__ == "__main__":
    cli()
