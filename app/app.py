import requests
import re

from datetime import datetime, timedelta
from json import dumps, load, loads

from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def api_health_check():
    message = 'Stev es el Grinch'
    return Response(message, status=200, mimetype='application/json')

@app.route('/api', methods=['GET'])
def api():
    data = {"Hi there": "This is a sample response"}
    return Response(dumps(data), status=200, mimetype='application/json')


@app.route('/api/<string:variable>', methods=['GET'])
def api_variable(variable):
    data = {"You entered variable:": variable}
    return Response(dumps(data), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)