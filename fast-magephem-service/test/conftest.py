# configuration for pytest

import os
import sys


testpath = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(testpath)
app_path = os.path.join(parentdir,'python')
if app_path not in sys.path:
    sys.path.insert(0,app_path) # take precedence over any other in path
