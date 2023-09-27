from collections import defaultdict, namedtuple
from datetime import datetime
import os
import json
import time

import numpy as np
import numpy.linalg
from scipy import interpolate, integrate
from scipy.stats import bootstrap
from scipy.optimize import nnls
import pandas as pd
from sklearn.linear_model import ElasticNet
from fancyimpute import SoftImpute, NuclearNormMinimization

import sqlalchemy
from sqlalchemy import text, bindparam, update, MetaData, Table
from sqlalchemy.orm import sessionmaker

from caws.database import CawsDatabase
from caws.features import CawsFeatureType

Prediction = namedtuple("Prediction", ["runtime", "energy"])

def split(df):
    gb = df.groupby(["block_id", 'pid'])
    return [gb.get_group(x) for x in gb.groups]

def sum_resources(df1, df2, columns=["perf_unhalted_core_cycles", "perf_unhalted_reference_cycles", "perf_llc_misses", "perf_instructions_retired", "cpu_freq_rel"]):
    df = pd.merge_asof(df1[columns], df2[columns], on="timestamp", direction="forward", tolerance=pd.Timedelta(3, unit="s")).fillna(0)
    for col in columns:
        df[col] = df[f"{col}_x"] + df[f"{col}_y"]
        df = df.drop([f"{col}_x", f"{col}_y"], axis=1)
    df = df.set_index("timestamp")
    return df

class EndpointModel:
    def __init__(self, tasks, static_power, energy_consumed, perf_counters, alpha=0.9):
        self.tasks = tasks
        self.static_power = static_power
        self.energy_consumed = energy_consumed
        self.alpha = alpha
        self.perf_counters = perf_counters

        self.regressions = {}

    def train(self, tasks, resources, energy, caws_df):
        resources = resources.dropna(subset=self.perf_counters)

        energy["timestamp"] = pd.to_datetime(energy['timestamp'])
        energy["power"] = energy["total_energy"] / energy["duration"]
        energy = energy.sort_values("timestamp", ignore_index=True)
        energy["block_id"] = energy[["run_id", "block_id"]].apply(lambda r: f"{r.run_id}.{r.block_id}", axis=1)
        self.energy_consumed += energy["total_energy"].sum()

        resources["timestamp"] = pd.to_datetime(resources['timestamp'])
        resources = resources.sort_values("timestamp", ignore_index=True)
        resources["block_id"] = resources[["run_id", "block_id"]].apply(lambda r: f"{r.run_id}.{r.block_id}", axis=1)

        tasks["block_id"] = tasks[["run_id", "block_id"]].apply(lambda r: f"{r.run_id}.{r.block_id}", axis=1)
        
        df_split = split(resources)
        df_split_clean = defaultdict(list)
        for i in range(len(df_split)):
            df_new = df_split[i][["timestamp", "perf_unhalted_core_cycles", "perf_unhalted_reference_cycles", "perf_instructions_retired", "perf_llc_misses"]].diff()
            df_new["timestamp"] = (df_new["timestamp"] / np.timedelta64(1, "s"))
            df_new = df_new.div(df_new["timestamp"], axis='index')
            df_new["cpu_freq_rel"] = (df_new["perf_unhalted_core_cycles"] + .000001) / (df_new["perf_unhalted_reference_cycles"] + .000001) 
            df_new["pid"] = df_split[i]["pid"]
            df_new["timestamp"] = df_split[i]["timestamp"]
            df_new = df_new.set_index("timestamp")
            df_new = df_new.dropna() # Drop first row of diff
            if len(df_new) > 0:
                df_split_clean[df_split[i]["block_id"].iloc[0]].append(df_new)

        process_preds = {}
        for block_id in df_split_clean.keys():
            df_split_clean[block_id] = sorted(df_split_clean[block_id], key=lambda x: x.index[-1] - x.index[0], reverse=True)
            df = df_split_clean[block_id][0]
            for i in range(1, len(df_split_clean[block_id])):
                df = sum_resources(df, df_split_clean[block_id][i])

            # TODO: Should I combine blocks/runs to create regression?
            df_combined = pd.merge_asof(energy[energy["block_id"] == block_id], df, on="timestamp", direction="backward")
            df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_combined = df_combined.dropna()
            regr = ElasticNet(random_state=0, positive=True)
            regr.fit(df_combined[self.perf_counters].values, df_combined["power"])
            df_combined["pred_power"] = regr.predict(df_combined[self.perf_counters].values)

            if self.static_power is None:
                self.static_power = regr.intercept_
            else:
                self.static_power = (self.alpha * self.static_power) + ((1-self.alpha) * regr.intercept_)

            for i in range(len(df_split_clean[block_id])):
                worker_df = df_split_clean[block_id][i]
                pid = worker_df["pid"].iloc[0]
                worker_df = pd.merge_asof(energy[energy["block_id"] == block_id], worker_df, on="timestamp", direction="backward").dropna()
                worker_df["pred_power"] = regr.predict(worker_df[self.perf_counters]) - regr.intercept_

                worker_df = pd.merge_ordered(
                    worker_df, tasks.loc[(tasks["pid"]==pid) & (tasks["block_id"]==block_id), "task_try_time_running"].rename("timestamp"),
                    on="timestamp"
                )
                worker_df = pd.merge_ordered(
                    worker_df, tasks.loc[(tasks["pid"]==pid) & (tasks["block_id"]==block_id), "task_try_time_running_ended"].rename("timestamp"),
                    on="timestamp"
                )

                worker_df = worker_df.set_index("timestamp")
                worker_df["pred_power"] = worker_df["pred_power"].interpolate(method="time")
                worker_df = worker_df.dropna(subset=["pred_power"])
                worker_df["energy"] = integrate.cumtrapz(worker_df["pred_power"], x=worker_df.index.astype(np.int64) / 10**9, initial=0)
                worker_df = worker_df[["energy"]]

                process_preds[(block_id, pid)] = worker_df

        tasks["task_try_time_running"] = pd.to_datetime(tasks['task_try_time_running'])
        tasks["task_try_time_running_ended"] = pd.to_datetime(tasks['task_try_time_running_ended'])
        tasks["running_duration"] = (tasks["task_try_time_running_ended"] - tasks["task_try_time_running"]) / np.timedelta64(1, 's')

        def calc_energy(row):
            try:
                return process_preds[(row.block_id, row.pid)].loc[row.task_try_time_running_ended, "energy"]- process_preds[(row.block_id, row.pid)].loc[row.task_try_time_running, "energy"]
            except:
                return None
        tasks["energy_consumed"]  = tasks.apply(calc_energy, axis=1)

        tasks = tasks[["task_id", "task_try_time_running", "running_duration", "energy_consumed"]]
        caws_df = caws_df[["caws_task_id", "funcx_task_id", "func_name", "time_began", "time_completed"]]

        tasks = pd.merge(tasks, caws_df, left_on="task_id", right_on="funcx_task_id", how="left")
        self.tasks = pd.concat([self.tasks, tasks])
        self.regressions = {}
        return tasks, self.static_power, self.energy_consumed

    def predict_func(self, func_name, features):
        categorical_features = tuple([v for v, t in features if t == CawsFeatureType.CATEGORICAL])
        continuous_features = [v for v, t in features if t == CawsFeatureType.CONTINUOUS]
        if func_name not in self.regressions:
            func_tasks = self.tasks[self.tasks["func_name"] == func_name]
            func_tasks = func_tasks.dropna(subset=["running_duration", "energy_consumed"])
            if len(func_tasks) == 0:
                return Prediction(None, None)

            # func_tasks["value"] = func_tasks['value'].fillna(1)
            func_tasks = func_tasks.sort_values("feature_id")
            func_tasks["features"] = func_tasks[['value', 'feature_type']].apply(tuple, axis=1)
            func_tasks = func_tasks.groupby("caws_task_id").agg(
                    running_duration = ("running_duration", 'first'),
                    energy_consumed = ("energy_consumed", 'first'),
                    feature_vals=('features', lambda s: [i[0] for i in s if i[1] == "CONTINUOUS"]), 
                    categories=('features', lambda s: tuple([i[0] for i in s if i[1] == "CATEGORICAL"]))
            )
            func_tasks = func_tasks[func_tasks["categories"] == categorical_features]
            func_tasks["avg_power"] = func_tasks["energy_consumed"] / func_tasks["running_duration"]

            data_matrix = np.stack(func_tasks["feature_vals"].values)
            (n, f) = data_matrix.shape
            data_matrix = np.c_[data_matrix, np.ones(n)]
            data_matrix = data_matrix[:, ~pd.isnull(data_matrix).any(axis=0)]
            y_matrix = func_tasks[["running_duration", "avg_power"]].to_numpy()
            w0, _ = nnls(data_matrix.astype(np.float64),  y_matrix[:, 0])
            w1, _ = nnls(data_matrix.astype(np.float64),  y_matrix[:, 1])
            w = np.stack([w0, w1], axis=1)

            self.regressions[(func_name, categorical_features)] = w
        else:
            w = self.regressions[(func_name, categorical_features)]

        continuous_features.append(1)
        y = np.array(continuous_features) @ w
        assert y[1] > 0, f"Predicted negative power, {func_name}, {continuous_features}, {w}, {y}"
        assert y[0] > 0, f"Predicted negative runtime, {func_name}, {continuous_features}, {w}, {y}"

        y[1] *= y[0]

        return Prediction(y[0], y[1])

    def predict_cold_start(self):
        cold_start_tasks = self.tasks[self.tasks["endpoint_status"] == "COLD"]
        if len(cold_start_tasks) == 0: # No information, or endpoint is always warm
            return 0
        total_time = (cold_start_tasks["time_completed"] - cold_start_tasks["time_began"]) / np.timedelta64(1, "s")
        wait_time = total_time - cold_start_tasks["running_duration"]
        return wait_time.mean()

    def predict_static_power(self):
        return self.static_power


class Predictor:
    def __init__(self, endpoints, caws_database_url):
        self.last_update_time = {}
        self.transfer_runtime_models = {} # Build transfer models on demand
        self.caws_database_url = caws_database_url
        self.endpoints = endpoints

        self.transfer_energy_models = {}
        directory = os.path.dirname(os.path.realpath(__file__))
        transfer_config_file = os.path.join(directory, "transfer_config.json")
        with open(transfer_config_file) as fp:
            transfer_energy_info = json.load(fp)

        n_switches = transfer_energy_info["defaults"]["num_switches"]
        edge_routers = transfer_energy_info["defaults"]["edge_routers"]
        hardware_models = transfer_energy_info["hardware_models"]
        for model in transfer_energy_info["transfer_models"]:
            key = (model["src"], model["dest"])
            core_routers = model["num_hops"] - edge_routers
            energy_per_bit = (n_switches * hardware_models["switch"])\
                             + (edge_routers * hardware_models["edge_router"])\
                             + (core_routers * hardware_models["core_router"])
            self.transfer_energy_models[key] = energy_per_bit
        
        self.embedding_matrix = None
            

    def start(self):
        self.eng = sqlalchemy.create_engine(self.caws_database_url) #TODO: Better method for this?
        self.Session = sessionmaker(bind=self.eng)
        with self.Session() as session:
            connection = session.connection()
            query = text("""SELECT caws_task.caws_task_id, func_name, funcx_task_id, endpoint_id, time_began, time_completed, endpoint_status, """
                """energy_consumed, running_duration, features.feature_id, features.feature_type, features.value """
                """FROM caws_task LEFT JOIN features ON caws_task.caws_task_id=features.caws_task_id WHERE ((task_status='COMPLETED') """
                """AND (endpoint_id in :endpoint_ids))""")
            query = query.bindparams(bindparam("endpoint_ids", [e.compute_endpoint_id for e in self.endpoints], expanding=True))
            func_to_tasks = pd.read_sql(query, connection)
            func_to_tasks = func_to_tasks.dropna(subset=["running_duration", "energy_consumed"])

            query = text("""SELECT * FROM caws_endpoint WHERE endpoint_id in :endpoint_ids""")
            query = query.bindparams(bindparam("endpoint_ids", [e.compute_endpoint_id for e in self.endpoints], expanding=True))
            endpoint_df = pd.read_sql(query, connection)
            endpoint_df = endpoint_df.set_index("endpoint_id")

            query = text("""SELECT * FROM transfer LIMIT 1000""")
            self.transfers = pd.read_sql(query, connection)
            self.transfers["runtime"] = (pd.to_datetime(self.transfers["time_completed"]) - pd.to_datetime(self.transfers["time_submit"])) / np.timedelta64(1, "s")

        self.endpoint_models = {}
        for endpoint in self.endpoints:
            tasks =  func_to_tasks[func_to_tasks["endpoint_id"] == endpoint.compute_endpoint_id]
            static_power = endpoint_df.loc[endpoint.compute_endpoint_id]["static_power"]
            energy_consumed = endpoint_df.loc[endpoint.compute_endpoint_id]["energy_consumed"]
            self.endpoint_models[endpoint.name] = EndpointModel(tasks, static_power, energy_consumed, endpoint.perf_counters)
    
    def create_embedding_table(self, func_to_tasks, n_examples=10):
        func_to_tasks = func_to_tasks.sort_values("feature_id")
        func_to_tasks = func_to_tasks.groupby("caws_task_id").agg(
            {'func_name': 'first', 'running_duration': 'first', 'energy_consumed': 'first', 'value': list})
        func_to_tasks["count"] = 1
        func_to_tasks = func_to_tasks.groupby("func_name").agg(
            {'value': 'first', "count": "count"}).sort_values("count", ascending=False)
        func_to_tasks = func_to_tasks.iloc[:n_examples].reset_index()
        
        embedding_matrix = np.zeros((2, len(self.endpoints), func_to_tasks.shape[0]))
        for i, endpoint_model in enumerate(self.endpoint_models.values()):
            result_df = func_to_tasks.apply(lambda r: list(endpoint_model.predict_func(r.func_name, [f for f in r.value if f is not None])), axis=1, result_type="expand")
            embedding_matrix[0, i, :] = result_df.iloc[:, 0].to_numpy()
            embedding_matrix[1, i, :] = result_df.iloc[:, 1].to_numpy()

        return embedding_matrix

    def update(self):
        for endpoint in self.endpoints:
            prev_query = self.last_update_time.get(endpoint.name, endpoint.start_time)
            tasks, resources, energy = endpoint.collect_monitoring_info(prev_query)
            if len(tasks) == 0:
                continue

            self.last_update_time[endpoint.name] = tasks["task_try_time_returned"].dropna().max().to_pydatetime()
            
            with self.Session() as session:
                connection = session.connection()
                query = text("""SELECT caws_task.caws_task_id, func_name, funcx_task_id, endpoint_id, time_began, time_completed, endpoint_status, """
                    """energy_consumed, running_duration, features.feature_id, features.feature_type, features.value """
                    """FROM caws_task LEFT JOIN features ON caws_task.caws_task_id=features.caws_task_id WHERE ((task_status='COMPLETED') """
                    """AND (endpoint_id = :endpoint_id)  AND (time_completed > :start_time)) """)
                query = query.bindparams(endpoint_id=endpoint.compute_endpoint_id, start_time=prev_query)
                caws_task = pd.read_sql(query, connection)

            tasks, static_power, energy_consumed = self.endpoint_models[endpoint.name].train(tasks, resources, energy, caws_task)
            tasks = tasks.dropna(subset=["caws_task_id"]) #TODO: Fix clock synchronization
            
            with self.Session() as session:
                values = tasks[["caws_task_id", "energy_consumed", "running_duration"]].to_dict('records')
                session.execute(update(CawsDatabase.CawsTask), values)
                session.commit()
                session.execute(update(CawsDatabase.CawsEndpoint), [{"endpoint_id": endpoint.compute_endpoint_id, "static_power": static_power, "energy_consumed": energy_consumed}])
                session.commit()

        with self.Session() as session:
            connection = session.connection()
            query = text("""SELECT * FROM transfer LIMIT 1000""")
            self.transfers = pd.read_sql(query, connection)
            self.transfers["runtime"] = (pd.to_datetime(self.transfers["time_completed"]) - pd.to_datetime(self.transfers["time_submit"])) / np.timedelta64(1, "s")
            # I don't delete the transfer models here

    def predict_execution(self, endpoint, task):
        # TODO: Figure out how to implement impute for missing values
        pred = self.endpoint_models[endpoint.name].predict_func(task.function_name, task.features)
        if pred.runtime is None or pred.energy is None:
            if self.embedding_matrix is None:
                with self.Session() as session:
                    connection = session.connection()
                    query = text("""SELECT caws_task.caws_task_id, func_name, funcx_task_id, endpoint_id, time_began, time_completed, endpoint_status, """
                        """energy_consumed, running_duration, features.feature_id, features.feature_type, features.value """
                        """FROM caws_task LEFT JOIN features ON caws_task.caws_task_id=features.caws_task_id WHERE ((task_status='COMPLETED') """
                        """AND (endpoint_id in :endpoint_ids))""")
                    query = query.bindparams(bindparam("endpoint_ids", [e.compute_endpoint_id for e in self.endpoints], expanding=True))
                    func_to_tasks = pd.read_sql(query, connection)
                    func_to_tasks = func_to_tasks.dropna(subset=["running_duration", "energy_consumed"])

                if len(func_to_tasks) == 0:
                    return None

                self.embedding_matrix = self.create_embedding_table(func_to_tasks)

            new_features = np.zeros((2, len(self.endpoints), 1))
            for i, model in enumerate(self.endpoint_models.values()):
                result = model.predict_func(task.function_name, task.features)
                new_features[0, i, 0] = result[0]
                new_features[1, i, 0] = result[1]

            self.embedding_matrix = np.concatenate((self.embedding_matrix, new_features), axis=2)

            runtime_filled = NuclearNormMinimization().fit_transform(self.embedding_matrix[0])
            energy_filled = NuclearNormMinimization().fit_transform(self.embedding_matrix[1])

            for i, other in enumerate(self.endpoints):
                if endpoint.name == other.name:
                    break
            result = Prediction(runtime_filled[i, -1], energy_filled[i, -1])

            # Trim embedding matrix again
            self.embedding_matrix = np.unique(self.embedding_matrix, axis=2)

            return result

        return pred

    def predict_transfer(self, src_endpoint_id, dst_endpoint_id, size, files):
        if src_endpoint_id == dst_endpoint_id:
            return Prediction(0, 0)

        if (src_endpoint_id, dst_endpoint_id) not in self.transfer_runtime_models:
            transfers = self.transfers[
                (self.transfers["src_endpoint_id"] == src_endpoint_id) \
                &  (self.transfers["dest_endpoint_id"] == dst_endpoint_id)]
            X = transfers[["bytes_transferred", "files_transferred"]].to_numpy()
            X = np.c_[X, np.ones(X.shape[0])]
            y = transfers["runtime"].to_numpy()
            w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.transfer_runtime_models[(src_endpoint_id, dst_endpoint_id)] = w

        else:
            w = self.transfer_runtime_models[(src_endpoint_id, dst_endpoint_id)]

        pred_runtime = np.array([size, files, 1]) @ w
        pred_energy = self.transfer_energy_models[(src_endpoint_id, dst_endpoint_id)] * size
        return Prediction(pred_runtime, pred_energy)        
        
    def predict_static_power(self, endpoint):
        return self.endpoint_models[endpoint.name].predict_static_power()

    def predict_cold_start(self, endpoint):
        return self.endpoint_models[endpoint.name].predict_cold_start()
