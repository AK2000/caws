import json
import datetime

import click
import pandas as pd
import sqlalchemy
from sqlalchemy import text

@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(readable=True),
    help="Location of experiment config.",
)
@click.option(
    "--limit", "-n",
    type=int,
    default=1000,
    help="Limit number of tasks"
)
@click.option(
    "--result_path", "-r",
    type=str,
    default="job_log.csv",
    help="Place to store results file"
)
def fetch_database(config, limit, result_path):
    config_obj = json.load(open(config, "r"))
    min_time = datetime.datetime.now() - datetime.timedelta(hours=24)
    query = text(("SELECT func_name, endpoint_id, running_duration, energy_consumed, llc_misses, instructions_retired, core_cycles, ref_cycles"
                 " FROM caws_task WHERE (task_status='COMPLETED') AND (energy_consumed IS NOT NULL)"
                 " AND time_began>:start_time"
                 " LIMIT :num_entries")).bindparams(start_time=min_time, num_entries=limit)
    
    engine = sqlalchemy.create_engine(config_obj["caws_monitoring_db"])
    conn = engine.connect()
    task_stats = pd.read_sql(query, conn)
    task_stats.to_csv(result_path, index=False)
    return

if __name__ == '__main__':
    fetch_database()