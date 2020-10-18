import requests
import re

from datetime import datetime, timedelta
from json import dumps, load, loads

from flask import Flask, render_template, Response, request, redirect, jsonify, send_from_directory, abort
from flask_cors import CORS


from services.pdf_services import analyze_text_pdf

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])

DOWNLOAD_DIRECTORY = ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def api_health_check():
    message = 'Nice tutorial!'
    return Response(message, status=200, mimetype='application/json')

@app.route('/test-textract', methods=['GET'])
def api_test_textract():
    output = processDocument()
    return Response(dumps(output), status=200, mimetype='application/json')

@app.route('/process-documents', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
	
    files = request.files.getlist('files[]')
	
    errors = {}
    success = False
    responsesDocs = []
	
    for file in files:		
        if file and allowed_file(file.filename):
            output = analyze_text_pdf(file.filename, file.read())  
            output['file_name'] = file.filename
            responsesDocs.append(output)
            success = True
        else:
            errors[file.filename] = 'File type is not allowed'
	
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        # add service textract responce
        resp = jsonify({'message' : 'Files successfully processed', 'responses_docs': responsesDocs})
        resp.status_code = 201
        resp.headers.add('Access-Control-Allow-Origin', '*')
        print('headers:: ', resp.headers)
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

@app.route('/get-document-with-boxes', methods=['POST'])
def get_document_with_boxes():
    doc_name = request.args.get('doc_name')
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, filename=doc_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)

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
