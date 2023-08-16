"""
code to run tests of fast magephem service
reads test data from test_data.json

usage:
    pytest from test dir

"""

import os

testpath = os.path.dirname(os.path.abspath(__file__))

import json
import pytest
from app import make_app
# ssdhas_shared is added to the path by importing from ssdl
from ssdhas_shared.python.test_utils import test_with_name_data

@pytest.fixture(scope='module')
def client():
    app = make_app(test_mode=True)
    with app.app.test_client() as c:
        yield c

# load tests from json file
# json file has this format:
#{
#<test-name>:
#    {
#    "endpoint": <api endpoint>,
#    "method": <method GET, PUT, POST, DELETE>,
#    "query-string": <optional, query string parameters as str w/o ?>,
#    "body": <optional, body object>,
#    "response-type": <text or json>
#    "response": <expected output text or json object>
#    },
#<next-test>...
#}

json_file = os.path.join(testpath,'test_data.json')
# load json file
with open(json_file,'r') as fid:
    json_data = json.load(fid)

test_with_name_data = pytest.mark.parametrize('name,data',json_data.items())(test_with_name_data)
