

from flask import Flask
from flask import Blueprint
from flask import request

endpoint_one = Blueprint('endpoint_1', __name__)
endpoint_two = Blueprint('endpoint_2',__name__)

@endpoint_one.route('/',methods=['GET'])
def main():

    #data = request.get_json()
    "Endpoint de prueba"
    return "hola mundo"

@endpoint_two.route('/request',methods=['POST'])
def endpoint_post():
    data = request.get_json()
    print(data)
    return {'ML_model_result':True}
