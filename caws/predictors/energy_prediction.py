import faust
from sklearn.linear_model import ElasticNet

class Task(faust.Record):
    try_id: int
    task_id: str
    run_id: str
    block_id: str 
    hostname: str
    pid: int
    task_try_time_running: str
    task_try_time_returned: str

class Status(faust.Record):
    try_id: int
    task_id: int
    task_status_name: str

class Energy(faust.Record):
    run_id: str
    block_id: str
    timestamp: str
    duration: int
    total_energy: float

class Resource(faust.Record):
    block_id: str
    run_id: str
    timestamp: str
    pid: int
    perf_unhalted_core_cycles = int
    perf_unhalted_reference_cycles = int
    perf_llc_misses = int
    perf_instructions_retired = int

async def add_process_power(resource):
    # Calculate the process power using the resource message and the trained
    # model
    # Can be written as a processer: https://faust.readthedocs.io/en/latest/userguide/streams.html#id2


@app.agent(value_type=Energy)
async def training_loop(energys, resources):
    new_stream = energy.window(timedelta=120).join(
        resources.groupby(Resource.run_id, Resource.block_id)
        ).window(timedelta=120, (Energy.run_id, Energy.block_id))

    async for energy in new_stream:
        # join the two stream on {block_id, run_id, timestamp}
        # train model
        # periodically (~ every 60 seconds)


@app.agent(value_type=Task)
async def task_predictor(tasks, statuses, predictions):
    async for order in orders:
        # join the the tasks and statuses on {task_id, try_id} to find 
        # time where task finished (status message status_name == running_ended)
        # join on the predictions stream on {run_id, block_id, pid}
        # linearly interpolate predictions stream to find power
        # push to caws_task table



