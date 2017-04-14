import requests
import pandas as pd
import numpy as np
import time
import os

from my_util import *

# Global vars
root_url = 'https://jobsense.sg/api' 
parse_url = root_url + '/get/job-title-parser/'
user, pwd = 'jobsense_cm', 'jzf\Hb"HgH~Y(aa(C@/Ye6$'

empty_df = pd.DataFrame({'domain': '', 'pri_func': '', 
                         'position': '', 'sec_func': ''}, index=[1])

def getDomain(parsed_title):
    domains = parsed_title['domains']
    return ', '.join(domains) if domains else ''

def getPriFunc(parsed_title):
    pri_func = parsed_title['pri_func']
    return str(pri_func[0]) if pri_func else '' 

def getPosition(parsed_title):
    position = parsed_title['position']
    return str(position[0]) if position else ''

def getSecFunc(parsed_title):
    sec_func = parsed_title['sec_func']
    return str(sec_func[0]) if sec_func else ''

def toStr(uni_ls):
    return [str(s) for s in uni_ls]

def get2Titles(response):
    tokens = response.json()['lexer_tokens']
    t1 = str(tokens[0]['value'])
    t2 = str(tokens[2]['value'])
    return t1, t2

def components(parsed_title):
    domain = getDomain(parsed_title); pri_func = getPriFunc(parsed_title)
    position = getPosition(parsed_title); sec_func = getSecFunc(parsed_title)
    
    return pd.DataFrame({'domain': domain, 'pri_func': pri_func, 
                         'position': position, 'sec_func': sec_func}, 
                        index=[1])

# two_titles = []
def getParts(response, job_title):
    j_obj = response.json();
    # if j_obj['exception'] == 'Invalid Job Title':
    #     print('{} is regarded as invalid by the parser'.format(job_title))
     	# global invalid_titles
     	# invalid_titles.append(job_title)

    keys = toStr(j_obj.keys())
    #     print(keys) # debug
    
    if ('output' in keys): 
        out = response.json()['output']
        if not out: 
            print('{} has no parsing output'.format(job_title))
            res = empty_df 
        else:
            if len(out) == 1:
                res = components(out[0]['title_1'])
            else:
                print('parsing {} gives > 1 title in output'.format(job_title))
                # global two_titles; two_titles.append(job_title)
                
                t1 = out[0]['title_1']; t2 = out[1]['title_2']
                c1, c2 = components(t1).iloc[0], components(t2).iloc[0]

                res = pd.DataFrame({'domain': intelJoin(c1.domain, c2.domain) , 
                                    'pri_func': intelJoin(c1.pri_func, c2.pri_func) , 
                                    'position': intelJoin(c1.position, c2.position), 
                                    'sec_func': intelJoin(c1.sec_func, c2.sec_func)},
                                    index=[1])
                # res = pd.concat([components(t1), components(t2)])
    else:
        res = empty_df
    
    res['title'] = job_title
    return res

global count; count = 0
def parse(job_title="software developer", verbose=False):
    global count
    if verbose: print('title %d' %count)
    count +=1
    response = requests.post(parse_url, auth=(user, pwd), 
                             json={"job_title":job_title, "verbose": verbose})
    
    #     can get parsed json-result from parser API
    if (response.status_code == 200):
        return getParts(response, job_title)
    else:
        print('cannot post parsing request of {} to parser API'.format(job_title))
        res = empty_df; res['title'] = job_title
        return res