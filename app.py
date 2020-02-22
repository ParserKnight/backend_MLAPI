from flask import Flask
from routing import base_api

app = Flask(__name__)
app.register_blueprint(base_api)

if __name__ == '__main__':
    app.run()

