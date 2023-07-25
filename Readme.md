# CAWS: Carbon Aware Workflow Scheduling
Caws is a python library/executor/service for running workflows across multiple sights in a carbon and energy aware fashion. The ultimate goal is to make the environmental impact of the computing jobs run transparent to the user, and provide incentive and automation to reduce that footprint.

## Installation
This repo can be used with the latest version of globus-compute-sdk. It can also schedule to any existing endpoint. However, to enable energy monitoring, the endpoints will need to be deployed using the forked version of [gobus-compute-endpoint](https://github.com/AK2000/funcX/tree/power_monitoring_new) as well as the forked version of [parsl](https://github.com/AK2000/parsl/tree/power-monitoring). In the configuration of the enpoints, you will need to enable monitoring, enable energy monitoring, and point the monitroing database to a location accessible from your personal compute (i.e wherever you are the scheduler from).

In the root folder of this repo, run:
```
$ pip install .
```

## Testing
To run the test suite use:
```
$ pytest 
```
