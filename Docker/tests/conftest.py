import os
import sys
from pathlib import Path
import pytest

app_path = os.path.join(os.path.dirname( __file__ ), '..')
print("app_pathe is ",app_path)
sys.path.insert(0, app_path)  # take precedence over any other in path

from app import create_app

# This is all straight from the flask testing tuturoal
# https://flask.palletsprojects.com/en/2.3.x/tutorial/tests/
@pytest.fixture
def app():
    #db_fd, db_path = tempfile.mkstemp()

    app = create_app(test_config="test_config")

    #with app.app_context():
    #    init_db()
    #    get_db().executescript(_data_sql)

    yield app

    #os.close(db_fd)
    #os.unlink(db_path)

@pytest.fixture
def client(app):
    return app.test_client()
