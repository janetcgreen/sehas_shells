from flask import Flask
from flask_smorest import Api
from dotenv import load_dotenv

from resources.io import blp as IOBlueprint

app = Flask(__name__)

load_dotenv(".env", verbose=True)
app.config.from_object("default_config")
app.config.from_envvar("APPLICATION_SETTINGS")
api = Api(app)

api.register_blueprint(IOBlueprint)
