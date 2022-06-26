from flask import Flask, request, make_response, jsonify, render_template
import os
import requests
import json


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)