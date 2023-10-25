"""
utilities for testing
recursive_equals - a recursive equality test that understands dicts, lists, numpy arrays, and dates
"""

import numpy as np
import datetime as dt
from pytest import approx

def recursive_equals(a,b,path='/'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray), '%s are not both numpy ndarrays ' % path
        assert len(a.dtype) == len(b.dtype), '%s have different dtype structure' % path
        if len(a.dtype) == 0: # simple array
            assert np.all_close(a,b,equal_nan=np.issubdtype(a.dtype,float)), 'arrays %s are not equal ' % path
            return True
        else: # structured array
            assert a.dtype.names != b.dtype.names, 'structured arrays %s have different keys ' % path
            for key in a.dtype.names:
                assert np.all_close(a[key],b[key],equal_nan=np.issubdtype(a[key].dtype,float)),'arrays %s/%s are not equal' % (path,key)
            return True
            
    if isinstance(a,dict):
        assert isinstance(b,dict),'%s are not both dicts ' % path
        assert sorted(a.keys())==sorted(b.keys()), 'dicts %s have different keys ' % path
        for key in a:
            assert recursive_equals(a[key],b[key],path+'/' + key)
        return True
    
    if isinstance(a,list):
        assert isinstance(b,list), '%s are not both lists ' % path
        assert len(a) == len(b), 'lists %s have different length ' % path
        for (i,ai) in enumerate(a):
            assert recursive_equals(ai,b[i],path+'/[' + str(i) + ']')
        return True

    if a is None:
        assert b is None, '%s are not both None' % path
        return True

    for base in (int,float,bool,str,dt.datetime,dt.timedelta):
        if np.issubdtype(type(a),base):
            assert np.issubdtype(type(b),base),'%s are not both %s' % (path,base.__name__)
            assert np.isscalar(a), 'a is not scalar %s' % path
            assert np.isscalar(b), 'b is not scalar %s' % path
            if base == float: # special case, handling nan and approximately equal
                if np.isnan(a):
                    assert np.isnan(b), 'a is nan, b is not at %s' % (path)
                else:
                    assert not np.isnan(b), 'b is nan, a is not at %s' % (path)
                    assert a==approx(b), 'a != b at %s' % path
            else: # all other classes require exact equals
                assert a==b, 'a != b at %s' % path
            return True

    assert False, 'Unable to compare %s types %s and %s' % (path,type(a),type(b))

# will need to decorate this
# e.g., test_with_name_data= pytest.mark.parametrize('name,data',json_data.items())(test_with_name_data)
def test_with_name_data(client,name,data):
    # test_with_name_data(client,name,data)
    # client - fixture that provides api test client, e.g., from flask app.test_client()
    # name - name of test
    # data - dict of data for test
    #    {
    #    "endpoint": <api endpoint>,
    #    "method": <method GET, PUT, POST, DELETE>,
    #    "query-string": <optional, query string parameters as str w/o ?>,
    #    "body": <optional, body object>,
    #    "response-type": <text or json>
    #    "response": <expected output text or json object>
    #    "status_code": <optional, expected numeric status code, defaults to 200>
    #    "api_root" : <optional, full path above data['endpoint']>
    #    },
    
    api_root = '/api'
    if 'api_root' in data:
        api_root = data['api_root']
        
    tmp_url = api_root+data['endpoint']
    
    options = {}
    
    if 'body' in data:
        options['json'] = data['body']
    
    if 'query-string' in data:
        tmp_url += '?' + data['query-string']
        
    if data['method'] == 'GET':
        result = client.get(tmp_url,**options)
    elif data['method'] == 'POST':
        result = client.post(tmp_url,**options)
    elif data['method'] == 'DELETE':
        result = client.delete(tmp_url,**options)
    elif data['method'] == 'PUT':
        result = client.put(tmp_url,**options)
    else:
        raise Exception('method %s not supported' % data['method'])
    
    status_code = 200
    if 'status_code' in data:
        status_code = data['status_code']
    assert result.status_code == status_code, 'Test client did not responsd with status %d, %s-> %s' % (status_code,tmp_url,result.get_data(as_text=True))
    
    if data['response-type'] == 'text':
        outputs = result.get_data(as_text=True)
    elif data['response-type'] == 'json':
        outputs = result.json
    else:
        raise Exception('Unknown response type "%s"' % data['response-type'])
    # do recursive equals to compare results from expected
    assert recursive_equals(data['response'],outputs,path=name)
