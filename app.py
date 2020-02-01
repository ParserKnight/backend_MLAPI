from flask import Flask
from routing import endpoint_one

app = Flask(__name__)
app.register_blueprint(endpoint_one)

if __name__ == '__main__':
    app.run()

