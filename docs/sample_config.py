from globus_compute_endpoint.endpoint.utils.config import Config
from globus_compute_endpoint.engines import GlobusComputeEngine
from parsl.providers import LocalProvider
from parsl.monitoring import MonitoringHub

config = Config(
    display_name=None,  # If None, defaults to the endpoint name
    executors=[
        GlobusComputeEngine(
            max_workers=1,
            provider=LocalProvider(
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
            ),
            energy_monitor="FakeNodeEnergyMonitor"
        )
    ],
    monitoring_hub=MonitoringHub(
        hub_address="localhost",
        hub_port=55055,
        monitoring_debug=True,
        resource_monitoring_interval=1,
        logging_endpoint="postgresql://<user>:<password>@<address>/monitoring"
    ),
)

# For now, visible_to must be a list of URNs for globus auth users or groups, e.g.:
# urn:globus:auth:identity:{user_uuid}
# urn:globus:groups:id:{group_uuid}
meta = {
    "name": "wsl-funcx-dev",
    "description": "",
    "organization": "",
    "department": "",
    "public": False,
    "visible_to": [],
}
