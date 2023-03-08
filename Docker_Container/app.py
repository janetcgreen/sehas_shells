import os

from flask import Flask
from flask_smorest import Api

from resources.prop import blp as PropsBlueprint
from resources.satellite import blp as SatelliteBlueprint

app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "SHELL REST API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
app.config["OUT_SCALE_FILENAME"] = os.eviron["OUT_SCALE_FILENAME"]
app.config["IN_SCALE_FILENAME"] = os.eviron["IN_SCALE_FILENAME"]
app.config["HDF5FILE"] = os.eviron["HDF5FILE"]

api = Api(app)

api.register_blueprint(PropsBlueprint)
api.register_blueprint(SatelliteBlueprint)
