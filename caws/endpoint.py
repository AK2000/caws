from enum import Enum
import requests
import json
import sqlalchemy
from sqlalchemy import text
import pandas as pd

from globus_compute_sdk import Executor

from caws.task import CawsTaskInfo
# from caws.predictors import RuntimePredictor, EnergyPredictor

electricitymaps_url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/history?zone=US-MIDW-MISO"
electricitymaps_headers = {
    "auth-token": "<INCLUDE>"
}

class EndpointState(Enum):
    DEAD: int = 0
    COLD: int = 1
    WARMING: int = 2
    WARM: int = 3

class Endpoint:
    name: str
    compute_endpoint_id: str
    transfer_endpoint_id: str
    state: EndpointState = EndpointState.COLD
    scheduled_tasks: set[str]
    running_tasks: list[CawsTaskInfo] = []

    # Location for fetching carbon information
    zone_id: str #i.e: US-MIDW-MISO for Chicago
    lat: str
    lon: str

    # Location for fetching electricity informatioin
    monitor_url: str

    def __init__(self, 
                 name, 
                 compute_id,
                 transfer_id,
                 lat=None,
                 lon=None,
                 state=EndpointState.COLD,
                 monitoring_avail: bool = False,
                 monitor_url="postgresql://parsl:parsl@192.5.87.108/monitoring",
                 model_file=None,
                 runtime_predictor=None,
                 queue_predictor=None,
                 energy_predictor=None):
        self.name = name
        self.compute_endpoint_id = compute_id
        self.transfer_endpoint_id = transfer_id
        self.state = state
        self.monitoring_avail = monitoring_avail
        self.monitor_url = monitor_url
        if self.monitoring_avail:
            self.monitoring_engine = sqlalchemy.create_engine(self.monitor_url)

        self.last_energy_timestamp = 0 # TODO: Change this to something more recent?
        self.last_carbon_time = None

        self.scheduled_tasks = []
        self.running_tasks = []

        self.runtime_predictor = runtime_predictor
        self.queue_predictor = queue_predictor
        self.energy_predictor = energy_predictor

        self.gce = Executor(endpoint_id=self.compute_endpoint_id)

    def collect_energy_use(self):
        if not self.monitoring_avail:
            return None

        with self.monitoring_engine.begin() as connection:
            result = connection.execute(text(f"SELECT hostname,end_time,total_energy FROM energy WHERE start_time >= {self.last_energy_timestamp}")).all()
            df = pd.DataFrame(result)
            self.last_energy_timestamp = df["end_time"].max()
            return df


    def collect_carbon_intensity(self):
        if not self.monitoring_avail:
            return None

        response = requests.get(electricitymaps_url, headers=electricitymaps_headers)
        self.latest_carbon_history = json.loads(response.text)
        #TODO: Deal with timezones in carbon history

    def predict_ETA(self, task):
        #TODO
        pass

    def predict_energy(self, task):
        #TODO
        pass

    def schedule(self, task):
        scheduled_tasks.add(task.task_id)
        
    def submit(self, task):
        if self.state == EndpointState.COLD:
            self.state = EndpointState.WARMING

        task.gc_future = self.gce.submit_to_registered_function(task.function_id, task.task_args, task.task_kwargs)
        task.gc_future.add_done_callback(task._update_caws_future)
        
    def poll(self):
        pass