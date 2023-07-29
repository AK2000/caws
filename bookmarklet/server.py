"""Flask server implementation."""
import psycopg

from flask import after_this_request
from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)
user = os.getenviron('CAWS_USER')
password = os.getenviron('CAWS_PASSWORD')
host = os.getenviron('CAWS_HOST')
dbname = 'monitoring'
tablename = 'energy'

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
    print(endpoint_id)

    with psycopg.connect(f'dbname={dbname} user={user} password={password} host={host}', row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(f'''SELECT *
                            FROM "{tablename}"
                            WHERE "{tablename}".run_id = \'{endpoint_id}\' '''
                       )
            data = cur.fetchall()
    if len(data) == 0:
        return jsonify({'run_id': endpoint_id, 'total_energy': 'N/A'})
    return jsonify(data[0])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, ssl_context='adhoc')
