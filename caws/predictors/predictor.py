from collections import defaultdict, namedtuple
from datetime import datetime

import numpy as np
import numpy.linalg
from scipy import interpolate, integrate
from scipy.stats import bootstrap

import pandas as pd

from sklearn.linear_model import ElasticNet

import sqlalchemy
from sqlalchemy import text, bindparam, update, MetaData, Table
from sqlalchemy.orm import sessionmaker


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
    def __init__(self, tasks, static_power, energy_consumed, alpha=0.9):
        self.tasks = tasks
        self.static_power = static_power
        self.energy_consumed = energy_consumed

        self.regressions = {}

    def train(self, tasks, resources, energy, caws_df):
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
            regr.fit(df_combined[["perf_instructions_retired", "perf_llc_misses"]].values, df_combined["power"])
            df_combined["pred_power"] = regr.predict(df_combined[["perf_instructions_retired", "perf_llc_misses"]].values)

            if self.static_power is None:
                self.static_power = regr.intercept_
            else:
                self.static_power = (alpha * self.static_power) + ((1-alpha) * regr.intercept_)

            for i in range(len(df_split_clean[block_id])):
                worker_df = df_split_clean[block_id][i]
                pid = worker_df["pid"].iloc[0]
                worker_df = pd.merge_asof(energy[energy["block_id"] == block_id], worker_df, on="timestamp", direction="backward").dropna()
                worker_df["pred_power"] = regr.predict(worker_df[["perf_instructions_retired", "perf_llc_misses"]]) - regr.intercept_

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
        caws_df = caws_df[["caws_task_id", "funcx_task_id", "func_name", "time_began"]]
        print(tasks)

        tasks = pd.merge(tasks, caws_df, left_on="task_id", right_on="funcx_task_id", how="left")
        self.tasks = self.tasks.append(tasks)
        self.regressions = {}
        return tasks, self.static_power, self.energy_consumed

    def predict_func(self, func_name, features):
        if func_name not in self.regressions:
            func_tasks = self.tasks[self.tasks["func_name"] == func_name]
            if len(func_tasks) == 0:
                return None

            func_tasks['value'].fillna(1)
            func_tasks = func_tasks.sort_values("feature_id")
            func_tasks = func_tasks.groupby("caws_task_id").agg(
                {'running_duration': 'first', 'energy_consumed': 'first', 'value': list})

            data_matrix = np.stack(func_tasks["value"].values)
            (n, f) = data_matrix.shape
            print(data_matrix)
            data_matrix = np.c_[data_matrix, np.ones(n)]
            y_matrix = func_tasks[["running_duration", "energy_consumed"]].to_numpy()
            print(y_matrix)

            w, _, _, _ = np.linalg.lstsq(data_matrix, y_matrix)
            self.regressions[func_name] = w
        else:
            w = self.regressions[func_name]

        y = features @ x
        return Prediction(y[0], y[1])

    def predict_cold_start(self):
        cold_start_tasks = self.tasks[self.tasks["endpoint_status"] == "COLD"]
        if len(cold_start_tasks) == 0: # No information, or endpoint is always warm
            return 0
        return (cold_start_tasks["task_try_time_running"] - cold_start_tasks["time_began"]).mean()

    def predict_static_power(self):
        return self.static_power


class Predictor:
    def __init__(self, endpoints, caws_database_url):
        self.eng = sqlalchemy.create_engine(caws_database_url) #TODO: Better method for this?
        self.Session = sessionmaker(bind=self.eng)
        with self.Session() as session:
            connection = session.connection()
            query = text("""SELECT caws_task.caws_task_id, func_name, funcx_task_id, endpoint_id, time_began, endpoint_status, """
                """energy_consumed, running_duration, features.feature_id, features.feature_type, features.value """
                """FROM caws_task LEFT JOIN features ON caws_task.caws_task_id=features.caws_task_id WHERE ((task_status='COMPLETED') """
                """AND (endpoint_id in :endpoint_ids))""")
            query = query.bindparams(bindparam("endpoint_ids", [e.compute_endpoint_id for e in endpoints], expanding=True))
            func_to_tasks = pd.read_sql(query, connection)

            query = text("""SELECT * FROM caws_endpoint WHERE endpoint_id in :endpoint_ids""")
            query = query.bindparams(bindparam("endpoint_ids", [e.compute_endpoint_id for e in endpoints], expanding=True))
            endpoint_df = pd.read_sql(query, connection)
            endpoint_df = endpoint_df.set_index("endpoint_id")

            query = text("""SELECT * FROM transfer LIMIT 1000""")
            self.transfers = pd.read_sql(query, connection)
            self.transfers["runtime"] = self.transfers["time_completed"] - self.transfers["time_submit"]

        self.endpoints = {}
        for endpoint in endpoints:
            tasks =  func_to_tasks[func_to_tasks["endpoint_id"] == endpoint.compute_endpoint_id]
            static_power = endpoint_df.loc[endpoint.compute_endpoint_id]["static_power"]
            energy_consumed = endpoint_df.loc[endpoint.compute_endpoint_id]["energy_consumed"]
            self.endpoints[endpoint.name] = EndpointModel(tasks, static_power, energy_consumed)
        
        self.last_update_time = {}
        self.transfer_models = {} # Build transfer models on demand

    def create_update(self, meta):
        def method(table, conn, keys, data_iter):
            sql_table = Table(table.name, meta, autoload=True)
            values = [dict(zip(keys, data)) for data in data_iter]
            print(values)
            update_stmt = update(sql_table)
            print(update_stmt)
            conn.execute(update_stmt, values)
        return method   

    def update(self, endpoint):
        prev_query = self.last_update_time.get(endpoint.name, endpoint.start_time)

        tasks, resources, energy = endpoint.collect_monitoring_info(prev_query)
        with self.Session() as session:
            connection = session.connection()
            query = text("""SELECT caws_task.caws_task_id, func_name, funcx_task_id, endpoint_id, time_began, endpoint_status, """
                """energy_consumed, running_duration, features.feature_id, features.feature_type, features.value """
                """FROM caws_task LEFT JOIN features ON caws_task.caws_task_id=features.caws_task_id WHERE ((task_status='COMPLETED') """
                """AND (endpoint_id = :endpoint_id)  AND (time_completed > :start_time)) """)
            query = query.bindparams(endpoint_id=endpoint.compute_endpoint_id, start_time=prev_query)
            caws_task = pd.read_sql(query, connection)

            query = text("""SELECT * FROM transfer LIMIT 1000""")
            self.transfers = pd.read_sql(query, connection)
            self.transfers["runtime"] = self.transfers["time_completed"] - self.transfers["time_submit"]
            # I don't delete the transfer models here

        tasks, static_power, energy_consumed = self.endpoints[endpoint.name].train(tasks, resources, energy, caws_task)
        print(tasks)

        with self.Session() as session:
            connection = session.connection()
            meta = MetaData(bind=connection)
            method = self.create_update(meta)
            tasks[["caws_task_id", "energy_consumed", "running_duration"]].to_sql("caws_tasks", connection, if_exists='append', method=method)

            sql_table = Table("caws_endpoint", meta, autoload=True)
            update_stmt = update(sql_table)
            session.execute(update_stmt, {"endpoint_id": endpoint.compute_endpoint_id, "static_power": static_power, "energy_consumed": energy_consumed})
        
        self.last_update_time[endpoint.name] = datetime.now()

    def predict_execution(self, endpoint, task):
        # TODO: Figure out how to implement impute for missing values
        pred = self.endpoints[endpoint.name].predict_func(task.function_name, task.features)
        if pred is None:
            # TODO: Implement low-rank matrix completion for missing values
            return None

        return pred

    def predict_transfer(self, src_endpoint, dst_endpoint, size, files):
        if (src_endpoint.transfer_endpoint_id, dst_endpoint.transfer_endpoint_id) in self.transfer_models:
            pred_runtime = np.array([size, files]) @ self.transfer_models[(src_endpoint.transfer_endpoint_id, dst_endpoint.transfer_endpoint_id)]

        transfers = self.transfers[
            (self.transfers["src_endpoint_id"] == src_endpoint.transfer_endpoint_id) \
            &  (self.transfers["dst_endpoint_id"] == dst_endpoint.transfer_endpoint_id)]
        
        X = transfers[["bytes_transferred", "files_transferred"]].to_numpy()
        y = transfers["runtime"]
        w, _, _, _ = np.linalg.lstsq(X, y)

        self.transfer_models[(src_endpoint.transfer_endpoint_id, dst_endpoint.transfer_endpoint_id)] = w
        pred_runtime = np.array([size, files]) @ w

        # TODO: Implement energy prediction
        return Prediction(pred_runtime, 0)        
        
    def predict_static_power(self, endpoint):
        return self.endpoints[endpoint.name].predict_static_power()

    def predict_cold_start(self, endpoint):
        return self.endpoints[endpoint.name].predict_cold_start()
