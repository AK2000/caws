import pandas as pd
from sklearn.linear_model import ElasticNet
import numpy as np
from scipy import interpolate, integrate
from scipy.stats import bootstrap
import sqlalchemy
from sqlalchemy import text
from collections import namedtuple

Prediction = namedtuple("Prediction", ["runtime", "energy"])

def split(df, columns):
        gb = df.groupby(columns)
        return [gb.get_group(x) for x in gb.groups]

def sum_resources(df1, df2, columns=["perf_unhalted_core_cycles", "perf_unhalted_reference_cycles", "perf_llc_misses", "perf_instructions_retired", "cpu_freq_rel"]):
    df = pd.merge_asof(df1[columns], df2[columns], on="timestamp", direction="forward")
    for col in columns:
        df[col] = df[f"{col}_x"] + df[f"{col}_y"]
        df = df.drop([f"{col}_x", f"{col}_y"], axis=1)
    df = df.set_index("timestamp")
    return df

class EndpointModel:
    def __init__(self, tasks, resources, energy, caws_df):
        self.train(tasks, resources, energy, caws_df)

    def train(self, tasks, resources, energy, caws_df):
        energy["timestamp"] = pd.to_datetime(energy['timestamp'])
        energy["power"] = energy["total_energy"] / energy["duration"]
        energy = energy.sort_values("timestamp", ignore_index=True)

        resources["timestamp"] = pd.to_datetime(resources['timestamp'])
        resources = resources.sort_values("timestamp", ignore_index=True)
        resources = resources.filter(regex='psutil_process_*', axis=1)
        resources["process_id"] = resources[["run_id", "block_id", "pid"]].apply('_'.join, axis=1)
        resources = resources.drop(["run_id", "block_id", "pid"], axis=1)
        
        df_split = split(resources)
        df_split_clean = []
        for i in range(len(df_split)):
            df_new = df_split[i].diff()
            df_new["timestamp"] = (df_new["timestamp"] / np.timedelta64(1, "s"))
            df_new = df_new.div(df_new["timestamp"], axis='index')
            df_new["cpu_freq_rel"] = (df_new["perf_unhalted_core_cycles"] + .000001) / (df_new["perf_unhalted_reference_cycles"] + .000001) 
            df_new["process_id"] = df_split[i]["process_id"]
            df_new["timestamp"] = df_split[i]["timestamp"]
            df_new = df_new.set_index("timestamp")
            df_split_clean.append(df_new)
        
        df = df_split_clean[0]
        for i in range(1, len(df_split_clean)):
            df = sum_resources(df, df_split_clean[i])
        df_combined = pd.merge_asof(energy, df, on="timestamp", direction="backward")
        
        df_combined = df_combined.dropna()
        regr = ElasticNet(random_state=0, positive=True)
        regr.fit(df_combined[["perf_instructions_retired", "perf_llc_misses"]], df_combined["power"])
        df_combined["pred_power"] = regr.predict(df_combined[["perf_instructions_retired", "perf_llc_misses"]])

        process_preds = {}
        for pid in process_preds:
            process_preds[pid] = pd.merge_ordered(
                process_preds[pid], trys.loc[trys["pid"] == pid, "timestamp"].rename(columns={"task_try_time_running": "timestamp"}),
                on="timestamp"
            )
            process_preds[pid] = pd.merge_ordered(
                process_preds[pid], trys.loc[trys["pid"] == pid, "timestamp"].rename(columns={"task_try_time_running_ended": "timestamp"}),
                on="timestamp"
            )
            process_preds[pid] = process_preds[pid].set_index("timestamp")
            process_preds[pid]["pred_power"] = process_preds[pid]["pred_power"].interpolate(method="time")
            process_preds[pid] = process_preds[pid].dropna(subset=["pred_power"])
            process_preds[pid]["energy"] = integrate.cumtrapz(process_preds[pid]["pred_power"], x=process_preds[pid].index.astype(np.int64) / 10**9, initial=0)
            process_preds[pid] = process_preds[pid][["timestamp", "energy"]]

        tasks["task_try_time_running"] = pd.to_datetime(tasks['task_try_time_running'])
        tasks["task_try_time_running_ended"] = pd.to_datetime(tasks['task_try_time_running_ended'])
        tasks["running_duration"] = tasks["task_try_time_running_ended"] - tasks["task_try_time_running"]
        tasks = tasks.set_index("task_id")

        def calc_energy(row):
            try:
                return process_preds[row.pid].loc[row.task_try_time_running_ended, "energy"]- process_preds[row.pid].loc[row.task_try_time_running, "energy"]
            except:
                return None
        tasks["energy"]  = tasks.apply(calc_energy, axis=1)

        self.energy_regr = regr
        tasks = tasks[["task_id", "task_try_time_running", "running_duration", "energy"]]
        self.tasks = pd.merge(tasks, caws_df, left_on="task_id", right_on="funcx_task_id", how="left")

    def update(self, tasks, resources, energy, caws_df):
        resources["timestamp"] = pd.to_datetime(resources['timestamp'])
        resources = resources.sort_values("timestamp", ignore_index=True)
        resources = resources.filter(regex='psutil_process_*', axis=1)
        resources["process_id"] = resources[["run_id", "block_id", "pid"]].apply('_'.join, axis=1)
        resources = resources.drop(["run_id", "block_id", "pid"], axis=1)
        
        df_split = split(resources)
        df_split_clean = []
        for i in range(len(df_split)):
            df_new = df_split[i].diff()
            df_new["timestamp"] = (df_new["timestamp"] / np.timedelta64(1, "s"))
            df_new = df_new.div(df_new["timestamp"], axis='index')
            df_new["cpu_freq_rel"] = (df_new["perf_unhalted_core_cycles"] + .000001) / (df_new["perf_unhalted_reference_cycles"] + .000001) 
            df_new["process_id"] = df_split[i]["process_id"]
            df_new["timestamp"] = df_split[i]["timestamp"]
            df_new = df_new.set_index("timestamp")
            df_split_clean.append(df_new)

        process_preds = {}
        for pid in process_preds:
            process_preds[pid] = pd.merge_ordered(
                process_preds[pid], trys.loc[trys["pid"] == pid, "timestamp"].rename(columns={"task_try_time_running": "timestamp"}),
                on="timestamp"
            )
            process_preds[pid] = pd.merge_ordered(
                process_preds[pid], trys.loc[trys["pid"] == pid, "timestamp"].rename(columns={"task_try_time_running_ended": "timestamp"}),
                on="timestamp"
            )
            process_preds[pid] = process_preds[pid].set_index("timestamp")
            process_preds[pid]["pred_power"] = process_preds[pid]["pred_power"].interpolate(method="time")
            process_preds[pid] = process_preds[pid].dropna(subset=["pred_power"])
            process_preds[pid]["energy"] = integrate.cumtrapz(process_preds[pid]["pred_power"], x=process_preds[pid].index.astype(np.int64) / 10**9, initial=0)
            process_preds[pid] = process_preds[pid][["timestamp", "energy"]]

        tasks["task_try_time_running"] = pd.to_datetime(tasks['task_try_time_running'])
        tasks["task_try_time_running_ended"] = pd.to_datetime(tasks['task_try_time_running_ended'])
        tasks["running_duration"] = tasks["task_try_time_running_ended"] - tasks["task_try_time_running"]
        tasks = tasks.set_index("task_id")

        def calc_energy(row):
            try:
                return process_preds[row.pid].loc[row.task_try_time_running_ended, "energy"]- process_preds[row.pid].loc[row.task_try_time_running, "energy"]
            except:
                return None
        tasks["energy"]  = tasks.apply(calc_energy, axis=1)
        tasks = tasks[["task_id", "task_try_time_running", "running_duration", "energy"]]
        self.tasks = pd.merge(tasks, caws_df, left_on="task_id", right_on="funcx_task_id", how="left")

    def predict_func(self, func_name, include_ci=False, confidence_level=0.9):
        func_tasks = self.tasks[self.tasks["func_name"] == func_name]
        runtime_mean = func_tasks["running_duration"].mean()
        runtime_std = func_tasks["running_duration"].std()
        energy_mean = func_tasks["energy"].mean()
        energy_std = func_tasks["energy"].std()

        if include_ci:
            runtime_mean_ci = bootstrap((func_tasks["running_duration"], ), np.mean, confidence_level=confidence_level).confidence_interval
            runtime_std = bootstrap((func_tasks["running_duration"], ), np.std, confidence_level=confidence_level).confidence_interval
            energy_mean_ci = bootstrap((func_tasks["energy"], ), np.mean, confidence_level=confidence_level).confidence_interval
            energy_std_ci = bootstrap((func_tasks["energy"], ), np.std, confidence_level=confidence_level).confidence_interval

            return (
                {
                    "runtime": (runtime_mean, runtime_std), 
                    "energy": (energy_mean, energy_std)
                }, 
                {
                    "runtime": (runtime_mean_ci, runtime_std), 
                    "energy": (energy_mean, energy_std)}
                )
            
        return Prediction(runtime_mean, energy_mean)

    def predict_cold_start(self):
        cold_start_tasks = self.tasks[self.tasks["endpoint_status"] == "COLD"]
        return (cold_start_tasks["task_try_time_running"] - cold_start_tasks["time_began"]).mean()

    def predict_static_power(self):
        return self.energy_regr.intercept_


class Predictor:
    def __init__(self, endpoints, caws_database_url):
        self.caws_db = sqlalchemy.create_engine(caws_database_url) #TODO: Better method for this?
        with self.caws_db.begin() as connection:
            query = text("SELECT func_name, funcx_task_id, endpoint_id, time_began, endpoint_status FROM caws_task").\
                        bindparams(endpoint_id=endpoint.compute_endpoint_id)
            func_to_tasks = pd.read_sql(query).dropna()

        self.endpoints = {}
        for endpoint in endpoints:
            tasks, resources, energy = endpoint.collect_monitoring_info()
            caws_df = func_to_tasks[func_to_tasks["endpoint_id"] == endpoint.compute_endpoint_id]
            self.endpoints[endpoint.name] = self.EndpointModel(self, tasks, resources, energy, caws_df)

    def predict(self, endpoint, task):
        return self.endpoints[endpoint.name].predict_func(task)
        
    def static_power(self, endpoint):
        return self.endpoints[endpoint.name].predict_static_power()

    def cold_start(self, endpoint):
        return self.endpoints[endpoint.name].predict_cold_start()