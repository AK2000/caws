from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.engines import GlobusComputeEngine
from globus_compute_endpoint.strategies import SimpleStrategy
import parsl
from parsl.addresses import address_by_interface
from parsl.providers import SlurmProvider
from parsl.monitoring import MonitoringHub

user_opts = {
    'midway': {
        'worker_init': 'source ~/.bashrc; conda activate funcx-dev',
        'scheduler_options': '#SBATCH --ntasks 1\n#SBATCH --cpus-per-task 48\n#SBATCH --exclusive',
    }
}

config = Config(
    display_name=None,  # If None, defaults to the endpoint name
    #detach_endpoint=False,
    executors=[
        GlobusComputeEngine(
            max_workers=48,
            provider=SlurmProvider(
                partition='caslake',
                account='pi-chard',
                launcher=parsl.launchers.SrunLauncher(),

                scheduler_options=user_opts['midway']['scheduler_options'],
                worker_init=user_opts['midway']['worker_init'],

                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,

                walltime='02:00:00',
            ),
            strategy=SimpleStrategy(
                # Shut down blocks idle for more that 30s
                max_idletime=300.0,
            ),
            energy_monitor="RaplCPUNodeEnergyMonitor"
        )
    ],
    monitoring_hub=MonitoringHub(
        hub_address=address_by_interface('bond0'),
        monitoring_debug=True,
        resource_monitoring_interval=1,
        logging_endpoint="postgresql://<user>:<password>@<address>/<database>"
    )
)

meta = {
    "name": "midway-funcx-dev",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}
