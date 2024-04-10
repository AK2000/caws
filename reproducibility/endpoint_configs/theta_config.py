from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.engines import GlobusComputeEngine
from globus_compute_endpoint.strategies import SimpleStrategy
import parsl
from parsl.addresses import address_by_interface
from parsl.providers import CobaltProvider
from parsl.monitoring import MonitoringHub

user_opts = {
    'theta': {
        'worker_init': 'module load conda; conda activate funcx-dev',
        'scheduler_options': '',
        'account': 'CSC249ADCD08'
    }
}

config = Config(
    display_name=None,  # If None, defaults to the endpoint name
    #detach_endpoint=False,
    executors=[
        GlobusComputeEngine(
            max_workers=64,
            cpu_affinity='block',
            provider=CobaltProvider(
                queue='debug-flat-quad',
                account='CSC249ADCD08',
                launcher=parsl.launchers.AprunLauncher(overrides='-d 64 --cc depth'),

                scheduler_options='',

                worker_init='module load conda; conda activate funcx-dev',

                nodes_per_block=1,
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,

                walltime='01:00:00',

                cmd_timeout=120,
            ),
            strategy=SimpleStrategy(
                # Shut down blocks idle for more that 30s
                max_idletime=45.0,
            ),
            energy_monitor="CrayNodeEnergyMonitor"
        )
    ],
    monitoring_hub=MonitoringHub(
        hub_address=address_by_interface('vlan2360'),
        monitoring_debug=False,
        resource_monitoring_enabled=True,
        resource_monitoring_interval=1,
        logging_endpoint="postgresql://<user>:<password>@<address>/<database>"
    )
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "theta-funcx-dev",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}