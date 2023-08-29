from flask import Flask
from flask_smorest import Api
from dotenv import load_dotenv

from shells_io import blp as IOBlueprint

def create_app(test_config=None):
    app = Flask(__name__)
    # The .env file has paths to the neural network files and
    # the IP address for connecting to the magephem service
    # NOTE: The magepehm address is currently the default docker bridge address
    # and will have to be changed to the NASA CCMC address for production

    # There is also a .env file in the test directory that is slightly different

    load_dotenv(".env", verbose=True)

    # There is a default_config.py and test_config.py that define things
    # like debug and test mode. The test code passes 'test_config'
    # and also sets some values like TESTL thats used to give mock
    # responses from the magepehm code

    if test_config==None:
        app.config.from_object('default_config')
    else:
        app.config.from_object(test_config)
    # JGREEN I don't think this is needed
    # app.config.from_envvar("APPLICATION_SETTINGS")

    # Uses flask-smorest which uses an enhanced blueprint
    # method for defining routes and marshmallow schema
    api = Api(app)

    # Register the blueprint that defines the shells inputs/outputs
    # and the /shells route imported from resources/io.py
    api.register_blueprint(IOBlueprint)

    return app