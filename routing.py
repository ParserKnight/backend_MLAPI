

from flask import Flask
from flask import Blueprint
from flask import request

endpoint_one = Blueprint('endpoint_1', __name__)
@endpoint_one.route('/',methods=('GET'))
def main():

    #data = request.get_json()
    "Endpoint de prueba"
    return "hola mundo"
