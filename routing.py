from flask import Flask
from flask import Blueprint
from flask import request
from handlers import errorHandler
from validators import Parser

base_api = Blueprint('base_api', __name__)

@base_api.route('/request/',methods=['POST'])
@errorHandler
def endpoint_post():
    """Endpoint for ML prediction"""

    data = request.get_json()
    validator = Parser(data)
    validator.validate()
    # var =  model.predict(data)
    # return {"result": var}
    #TODO Save in database to constraint repetition
    
    return {'result':True}

