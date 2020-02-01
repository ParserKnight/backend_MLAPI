from flask import Flask
from routing import endpoint_one, endpoint_two

app = Flask(__name__)
app.register_blueprint(endpoint_one)
app.register_blueprint(endpoint_two)

if __name__ == '__main__':
    app.run()

