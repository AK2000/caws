"""Flask server implementation."""
import pandas as pd
import psycopg
import os

from flask import after_this_request
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)
user = os.environ['CAWS_USER']
password = os.environ['CAWS_PASSWORD']
host = os.environ['CAWS_HOST']
dbname = 'monitoring'

def get_kwh(microjoules):
    return round(microjoules * 10**-6 / 3600, 7)

@app.route("/caws", methods=["GET"])
def get_energy() -> bytes:
    """REST request to return proxy/ies to the user.
    
    Returns (bytes):
        The pickled dictionary of all proxies or a single pickled proxy if \
            database name to return is provided.
    """
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    endpoint_id = request.args.get('endpoint')

    with psycopg.connect(f'dbname={dbname} user={user} password={password} host={host}', row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(f'''SELECT *
                            FROM "energy"
                            INNER JOIN "caws_task"
                            ON "energy".run_id = "caws_task".endpoint_id
                            WHERE "energy".run_id = \'{endpoint_id}\' '''
                       ) # FULL JOIN so we can get energy data of endpoint even if one of the table parameters is missing
            data = cur.fetchall()
    if len(data) == 0:
        return jsonify({'run_id': endpoint_id, 'energy_consumed': 'N/A', 'total_energy': 'N/A'})
    
    df = pd.DataFrame(data)
    total_energy = df['total_energy'].sum()
    energy_consumed = df['energy_consumed'].sum()
    
    return jsonify({'run_id': endpoint_id, 'energy_consumed': get_kwh(energy_consumed), 'total_energy': get_kwh(total_energy)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001, ssl_context='adhoc')
