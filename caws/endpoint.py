from enum import Enum, IntEnum
import requests
import json
import sqlalchemy
from sqlalchemy import select, or_, and_
import pandas as pd
import os
import datetime
from importlib.resources import files

from parsl.monitoring.db_manager import Database as mdb
from globus_compute_sdk import Executor

from caws.utils import client
from caws.task import CawsTaskInfo
# from caws.predictors import RuntimePredictor, EnergyPredictor
electricitymaps_url = "https://api-access.electricitymaps.com/free-tier/carbon-intensity/history"
geolite2_db = os.path.join(os.path.dirname(__file__), 'data', 'GeoLite2-City', 'GeoLite2-City.mmdb')

class EndpointState(IntEnum):
    DEAD: int = 0
    COLD: int = 1
    WARMING: int = 2
    WARM: int = 3

class EndpointOperations(Enum):
    TRANSFER: str = "transfer"
    COMPUTE: str = "compute"

class Endpoint:
    name: str
    compute_endpoint_id: str
    transfer_endpoint_id: str
    state: EndpointState = EndpointState.COLD
    scheduled_tasks: set[str]
    running_tasks: list[CawsTaskInfo] = []
    operations: list[EndpointOperations] = []

    # Location for fetching carbon information
    zone_id: str #i.e: US-MIDW-MISO for Chicago
    lat: str
    lon: str

    # Location for fetching electricity informatioin
    monitor_url: str

    def __init__(self, 
                 name, 
                 compute_id,
                 transfer_id=None,
                 endpoint_path:str = "~/", # TODO: Figure out resolve use for default path
                 local_path:str = None,
                 monitoring_avail: bool = True,
                 monitor_url=None,
                 monitor_carbon: bool = False,
                 lat=None,
                 lon=None,
                 zone_id=None,
                 slots_per_block=1,
                 min_blocks=0,
                 max_blocks=1,
                 parallelism=0.5,
                 shutdown_time=10):
                 
        self.name = name
        
        self.compute_endpoint_id = compute_id
        if self.compute_endpoint_id:
            self.operations.append(EndpointOperations.COMPUTE)

        self.transfer_endpoint_id = transfer_id
        if self.transfer_endpoint_id:
            self.operations.append(EndpointOperations.TRANSFER)
        self.endpoint_path = endpoint_path
        self.local_path = local_path if local_path is not None else endpoint_path

        self.monitoring_avail = monitoring_avail
        self.monitor_url = monitor_url
        if self.monitoring_avail:
            try:
                if self.monitor_url is None:
                    self.monitor_url = os.environ["ENDPOINT_MONITOR_DEFAULT"]
                self.monitoring_engine = sqlalchemy.create_engine(self.monitor_url)
            except:
                self.monitoring_avail = False
        else:
            self.monitoring_avail = False

        self.last_energy_timestamp = None # TODO: Change this to something more recent?
        self.last_carbon_timestamp = None
        
        self.scheduled_tasks = set()
        self.running_tasks = set()

        # Fetch state and status
        self.poll()
        self.metadata = client.get_endpoint_metadata(self.compute_endpoint_id)

        self.gce = Executor(endpoint_id=self.compute_endpoint_id,
                            monitoring=self.monitoring_avail)

        self.monitor_carbon = monitor_carbon
        if self.monitor_carbon:
            self.lat = lat
            self.lon = lon
            self.zone_id = zone_id

            if self.lat is None or self.lon is None or self.zone_id is None:
                # This is expensive. Should only have to do this once for all endpoints
                # But need to figure out how to close connection afterward
                import geoip2.database

                with geoip2.database.Reader(geolite2_db) as reader:
                    response = reader.city(self.metadata['ip_address'])
                    self.lat = response.location.latitude
                    self.lon = response.location.longitude

                self.electricitymaps_header = { 'auth-token': os.environ["ELECTRICITY_MAPS_TOKEN"] }
        
        self.slots_per_block = slots_per_block
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.parallelism = parallelism

        self.active_slots = self.slots_per_block * self.min_blocks
        self.active_tasks = 0

        if self.active_slots > 0:
            self.state = EndpointState.WARM
        else:
            self.state = EndpointState.COLD
        
        self.start_time = datetime.datetime.now()

    def collect_monitoring_info(self, prev_timestamp = None):
        if not self.monitoring_avail:
            return None

        if prev_timestamp == None:
            prev_timestamp = self.start_time

        with self.monitoring_engine.begin() as conn:
            run_ids = list(conn.execute(select(mdb.Workflow.run_id).where(
                and_(
                    mdb.Workflow.workflow_name == self.compute_endpoint_id,
                    or_(
                        mdb.Workflow.time_completed > prev_timestamp,
                        mdb.Workflow.time_completed == None
                    )
                )
            )).all())
            task_run_ids_or = [mdb.Try.run_id == run_id[0] for run_id in run_ids]
            task_df = pd.read_sql(select(mdb.Try).where(and_(or_(*task_run_ids_or), mdb.Try.task_try_time_launched > prev_timestamp)), conn)

            end_times = pd.read_sql(select(mdb.Status).where(and_(mdb.Status.task_status_name == "running_ended", mdb.Status.timestamp > prev_timestamp)), conn)
            task_df = pd.merge(task_df, end_times, on="task_id")
            task_df = task_df.rename(columns={"timestamp": "task_try_time_running_ended"})
            task_df["task_try_time_running"] = pd.to_datetime(task_df["task_try_time_running"])
            task_df["task_try_time_running_ended"] = pd.to_datetime(task_df["task_try_time_running_ended"])

            resource_run_ids_or = [mdb.Resource.run_id == run_id[0] for run_id in run_ids]
            resources_df = pd.read_sql(select(mdb.Resource).where(and_(or_(*resource_run_ids_or), mdb.Resource.timestamp > prev_timestamp)), conn)

            energy_run_ids_or = [mdb.Energy.run_id == run_id[0] for run_id in run_ids]
            energy_df = pd.read_sql(select(mdb.Energy).where(and_(or_(*energy_run_ids_or), mdb.Energy.timestamp > prev_timestamp)), conn)

        return task_df, resources_df, energy_df


    def collect_carbon_intensity(self):
        if not self.monitor_carbon:
            return None

        now = datetime.datetime.now()
        if self.last_carbon_timestamp is not None and now - self.last_carbon_timestamp < datetime.datetime.day:
            return self.latest_carbon_history

        payload = {
            "lat": self.lat,
            "lon": self.lon,
            "zone": self.zone_id
        }
        response = requests.get(electricitymaps_url, 
                                params=payload,
                                headers=self.electricitymaps_header)
        self.latest_carbon_history = json.loads(response.text)
        for hour in self.latest_carbon_history["history"]:
            hour["datetime"] = datetime.datetime.strptime(hour["datetime"],
                                                          '%Y-%m-%dT%H:%M:%S.%fz')
            hour["datetime"] = hour["datetime"]\
                                .replace(tzinfo=datetime.timezone.utc)\
                                .astimezone(None)\
                                .replace(tzinfo=None)

        self.last_carbon_timestamp = now

        #TODO: Deal with timezones in carbon history
        return self.latest_carbon_history

    def schedule(self, task):
        self.scheduled_tasks.add(task.task_id)
    
    def discard(self, task):
        self.scheduled_tasks.discard(task.task_id)
        
    def submit(self, task):
        if self.state == EndpointState.COLD:
            self.state = EndpointState.WARMING

        self.scheduled_tasks.discard(task.task_id)
        task.endpoint = self
        task.gc_future = self.gce.submit(task.func, *task.task_args, **task.task_kwargs)
        self.running_tasks.add(task.task_id)
        self.active_tasks += 1

    def task_finished(self, task):
        self.running_tasks.remove(task.task_id)
        self.active_tasks -= 1

    def poll(self) -> EndpointState:
        status = client.get_endpoint_status(self.compute_endpoint_id)
        self.status = status

        if status["status"] != "online":
            self.state = EndpointState.DEAD
        elif len(status["details"]["active_managers"]) > 0 and self.state != EndpointState.WARM:
            self.state = EndpointState.WARM
        elif len(status["details"]["active_managers"]) == 0 and self.state != EndpointState.WARMING:
            self.state = EndpointState.COLD

        return self.state

