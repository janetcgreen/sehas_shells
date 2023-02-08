"""
code to run tests of fast magephem service
reads test data from test_data.json

usage:
    python run_tests.py
    python run_tests.py -
     connects to http://localhost:23761/api
    python run_tests.py http://host.name:port/path/to/api
     port is optional, defaults to 23761
    additional arguments:        
     --use-proxy do not disable proxy (default disables proxy for internal testing)
     --file=<json_file_name> specify alternate json file (default is test_data.json)
     --single-test=<test_name> only perform one test in the json file
"""

import os
import sys
import json
import numpy as np
import datetime as dt
import requests
import traceback

def recursive_equals(a,b,path='/'):
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray):
            print('%s are not both numpy ndarrays ' % path)
            return False
        if len(a.dtype) != len(b.dtype):
            print('%s have different dtype structure' % path)
            return False
        if len(a.dtype) == 0: # simple array
            if np.array_equal(a,b,equal_nan=np.issubdtype(a.dtype,float)):
                return True
            else:
                print('arrays %s are not equal ' % path)
                return False
        else: # structured array
            if a.dtype.names != b.dtype.names:
                print('structured arrays %s have different keys ' % path)
                return False
            for key in a.dtype.names:
                if not np.array_equal(a[key],b[key],equal_nan=np.issubdtype(a[key].dtype,float)):
                    print(key,a[key].dtype,np.issubdtype(a[key].dtype,float))
                    print('arrays %s/%s are not equal' % (path,key))
                    return False
            return True
            
    if isinstance(a,dict):
        if not isinstance(b,dict):
            print('%s are not both dicts ' % path)
            return False
        if not recursive_equals(sorted(a.keys()),sorted(b.keys()),path):
            print('dicts %s have different keys ' % path)
            return False
        for key in a:
            if not recursive_equals(a[key],b[key],path+'/' + key):
                return False
        return True
    
    if isinstance(a,list):
        if not isinstance(b,list):
            print('%s are not both lists ' % path)
            return False
        if len(a) != len(b):
            print('lists %s have different length ' % path)
            return False
        for (i,ai) in enumerate(a):
            if not recursive_equals(ai,b[i],path+'/[' + str(i) + ']'):
                return False
        return True

    if a is None:
        if b is None:
            return True
        else:
            print('%s are not both None' % path)
            return False

    for base in (int,float,bool,str,dt.datetime,dt.timedelta):
        if np.issubdtype(type(a),base):
            if not np.issubdtype(type(b),base):
                print('%s are not both %s' % (path,base.__name__))
                return False
            if (not np.isscalar(a)) or not np.isscalar(b):
                return False
            if (base == float) and np.isclose(a,b,equal_nan=True):
                return True
            if a == b:
                return True
            else:
                print('%s are unequal' % path)
                return False        
    
    print('Unable to compare %s types %s and %s' % (path,type(a),type(b)))
    return False


if __name__ == '__main__':
    url = None
    port = 23761 # assigned port for fast magephem service
    if (len(sys.argv)>1) and (sys.argv[1] != '-'):
        url = sys.argv[1]
        url = url.rstrip('/') # remove trailing /
        parts = url.split('/')
        if ':' not in parts[0]: # append port
            parts[0] = '%s:%d' % (parts[0],port)
        url = '/'.join(parts)
    else:
        url = 'http://localhost:%d/api' % port
        
    NoProxy = True
    DataFile = 'test_data.json'
    SingleTest = False
    for arg in sys.argv[2:]:
        if arg.startswith('--file='):
            DataFile = arg.split('=')[1]
        elif arg.startswith('--single-test='):
            SingleTest = arg.split('=')[1]
        elif arg.lower() == '--use-proxy':
            NoProxy = False
        else:
            raise Exception('Unkonwn argument:%s' % arg)
            
    if NoProxy:
        proxies = {
          "http": None,
          "https": None,
        }        
    else:
        proxies = {}
    
    # find json file in same directory as this file
    path = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(path,'test_data.json')
    # load json file
    with open(json_file,'r') as fid:
        json_data = json.load(fid)
        
    # run tests
    print('%s\nTesting on URL %s' % ('='*40,url))
    for (name,data) in json_data.items():
        if SingleTest and (name != SingleTest):
            continue # skip others in single test mode
        print('\n\n%s\nTEST:%s' % ('-'*40,name))
        try:
            # run the test
            if data['method'] == 'POST':
                result = requests.post(url+data['endpoint'],json=data['inputs'],proxies=proxies)
            else:
                raise Exception('method %s not supported' % data['method'])
            outputs = result.json()
            # do recursive equals to compare results from expected
            if recursive_equals(data['outputs'],outputs):
                print('RESULT:PASS %s' % name)
            else:
                raise Exception('result not equal to expected outputs')
        except Exception as e:
            # there was a failure. Report
            print('RESULT:FAIL %s, %s' % (name,str(e)))
            print(traceback.format_exc()) # print detailed failure
            print(data['outputs'],outputs)
