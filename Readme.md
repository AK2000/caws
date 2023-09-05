# CAWS: Carbon Aware Workflow Scheduling
Caws is a python library/executor/service for running workflows across multiple sights in a carbon and energy aware fashion. The ultimate goal is to make the environmental impact of the computing jobs run transparent to the user, and provide incentive and automation to reduce that footprint.

## Setup
This repo can be used with the latest version of globus-compute-sdk. It can also schedule to any existing endpoint. However, to enable energy monitoring, the endpoints will need to be deployed using the forked version of [gobus-compute-endpoint](https://github.com/AK2000/funcX/tree/power_monitoring_new) as well as the forked version of [parsl](https://github.com/AK2000/parsl/tree/power-monitoring). In the configuration of the enpoints, you will need to enable monitoring, enable energy monitoring, and point the monitroing database to a location accessible from your personal compute (i.e wherever you are the scheduler from).

### Endpoints
First you have to setup a globus compute endpoint with the correct forks of all the repositories. The general globus compute documentation is [here](https://globus-compute.readthedocs.io/en/2.1.0/quickstart.html), but below I provide modified instructions that show how to configure an endpoint for use with CAWS. 

On a system where you want to run a compute endpoint, use the following commands to install globus-compute-endpoint and create an endpoint:
```
git clone git@github.com:AK2000/funcX.git
cd funcX/compute_endpoint
pip install . # Install globus compute endpoint

globus-compute-endpoint configure <ENDPOINT_NAME>
```
Then you have to replace the config.yaml file in `~/.globus_compute/<ENDPOINT_NAME>/config.yaml` with an appropriate `config.py` file to correctly configure the endpoint. In the configuration, be sure to use the `GlobusComputeEngine` and specify an `energy_monitor`. This is a system specific class that tells the endpoint how it can read the total energy that is being used by a node.

Monitoring must aslo be enabled in the endpoint configration. Monitoring can be enabled in configuration of the endpoint using the monitoring infrastructure derived from parsl:

```
from parsl.monitoring import MonitoringHub
...
config = Config(...,
                monitoring_hub=MonitoringHub(
                        hub_address="localhost",
                        hub_port=55055,
                        monitoring_debug=True,
                        resource_monitoring_interval=1,
                        logging_endpoint="postgresql://<user>:<password>@<address>/monitoring"),
)
```
The `logging_endpoint` is a relational database that stores resource and task monitoring information and serves as a source for the CAWS client.

A sample configuration with both monitoring enabled and the correct executor configuration is in `docs/sample_config.json`.

After the Globus Compute Endpoint is properly configured, start the endpoint:
```
globus-compute-endpoint start <ENDPOINT_NAME>
```

### Host

Finally, to install this repo with it's dependencies and the experiments, run:
```
$ pip install .
```
Be sure to also setup the environment variable. Alternatively, you can pass it as an argument with every executor that you create. 
```
export ENDPOINT_MONITOR_DEFAULT=<DATABASE_URI>
```

## Testing
To run the test suite use:
```
$ pytest --endpoint_id <COMPUTE_ID>
```

## Experiments

## Bookmarklet
