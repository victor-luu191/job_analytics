from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from time import time

# Given index_of_users and index_of_items,
## build user-item matrix from a df with columns (user_col, item_col, rating_col) containing triples (uid, item_id, rating)
def buildUserItemMat(df, index_of_users, index_of_items, user_col = 'uid', item_col = 'item_id', rating_col = 'rating'):
    
    n_user, n_item = len(index_of_users.keys()), len(index_of_items.keys())
    print('# users in index: %d' %n_user)
    print('# items in index: %d' %n_item)

    print('Mapping user ids to internal user indices...')
    row_ind = list(df.apply(lambda r: index_of_users[r[user_col]], axis=1))
    print('Mapping item ids to internal item indices...')
    col_ind = list(df.apply(lambda r: index_of_items[r[item_col]], axis=1))
    ratings = list(df[rating_col])
    
    user_item_mat = sp.csr_matrix((ratings, (row_ind, col_ind)), shape=(n_user, n_item))
    print('User-Item matrix built')
    return user_item_mat

def mkUserIndex(df, user_col='user_id'):
    user_ids = np.unique(df[user_col])
    index_of_users = { user_ids[i]:i for i in range(len(user_ids)) }
    return index_of_users

def mkItemIndex(df, item_col = 'item_id'):
    
    item_ids = np.unique(df[item_col])
    index_of_items = { item_ids[i]:i for i in range(len(item_ids))}
    return index_of_items

def printInfo(sp_mat):
    print ('Dims of matrix: {}'.format(sp_mat.shape))
    print('# non-zero entries: %d' %sp_mat.nnz)
    print('Max entry: %d' %sp_mat.max())

def toDate(date_str):
    date_format = '%Y-%m-%d'
    return datetime.strptime(date_str, date_format)

def calDuration(app_df):
    user_apply_job_cases = app_df[['uid', 'job_title', 'apply_date']].groupby(by=['uid', 'job_title'])
    print('Finished grouping by user-job cases. Totally we have {} cases.'.format(len(user_apply_job_cases)))
    
    t0 = time()
    print('Calculating duration of application for each case...')
    res = user_apply_job_cases['apply_date'].agg([min, max, 'nunique'])
    res['duration'] = np.subtract(map(toDate, res['max']), map(toDate, res['min']))
    print('Done after: {}s'.format(time()-t0))
    # Get duration in days
    res['duration'] = map(lambda x: x.days, res['duration'])
    res.rename(columns={'min': 'first_apply_date', 'max': 'last_apply_date', 'duration': 'total_duration_in_day', 
                        'nunique': 'n_active_day'}, inplace=True)
    return res

def loglog(series, xl, yl):
    
    h = series.value_counts()
    uniq_vals, counts = h.keys(), h.values
    
    plt.scatter(x=uniq_vals, y=counts)
    plt.loglog()
    
    plt.xlabel(xl), plt.ylabel(yl)
    plt.xlim(min(uniq_vals), max(uniq_vals))
    plt.grid(True)
#     plt.show()
#     return fig

# change dots in column names to dashes
def dot2dash(df):
    return df.rename(columns= {name: name.replace('.', '_') for name in df.columns})