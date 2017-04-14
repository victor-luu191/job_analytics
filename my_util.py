import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from string import punctuation
from matplotlib.ticker import ScalarFormatter
from difflib import Differ

## Sorting ===================================================
# dat can be any data type having sort_values()
def sortDown(dat):
    return dat.sort_values(ascending=False)

## R to Python ===================================================
def dot2dash(names):
    return map(lambda s: s.replace('.', '_'), names)

## Stats funcs ===================================================
def quantile(a, qs = [0, 25, 50, 75, 100], dec=1):
    values = np.around(np.percentile(a, q=qs), decimals=dec)
    df = pd.DataFrame({'min': values[0], '25%': values[1], '50% (median)': values[2], '75%': values[3], 'max': values[4]}, 
                      index=[0])
    return df[['min', '25%', '50% (median)', '75%', 'max']]

def normalize(arr):
    s = sum(arr)
    return [arr[i]/float(s) for i in range(len(arr))]

## DM funcs ===================================================
def mkPartition(n_instances, p=80):
    np.random.seed(123)
    
    train_size = n_instances*p/100
    idx = range(n_instances)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[: train_size], idx[train_size:]
    return train_idx, test_idx


## Text processing ===================================================
def n_word(s):
    return len(s.split())

def punc2space(d):
    r = re.compile(r'[{}]'.format(punctuation))
    new_doc = r.sub(' ', d)
    return new_doc

def n_words_in_doc(d):
    words = re.split(r'[^0-9A-Za-z]+', d)
    words = [w for w in words if w !='']
    return len(words)

# keep only words (including numbers) while removing all symbols, punctuations
def words_in_doc(d):
    words = re.split(r'[^0-9A-Za-z]+', d)
    words = [w for w in words if w !='']
    return words

## String funcs ===================================================
# def paste(strings):
#     return ','.join(strings)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False        

def found(keys, s):
    """
    return True if s contains One of the keys, False otherwise
    """
    return any([(k in s) for k in keys])

def clean(s):
    return s.replace(' ', '_').replace('/', '').replace('and', '_').replace(' ', '')

def joinEntries(ls):
    return ', '.join(ls)

def toStr(uni_ls):
    return [str(s) for s in uni_ls]

def camelCase(s):
    return s.title()

def tagChanges(s1, s2):
    "Adds <b></b> tags to words that are changed"
    l1 = s1.split(' ')
    l2 = s2.split(' ')
    dif = list(Differ().compare(l1, l2))
    return " ".join(['<b>'+i[2:]+'</b>' if i[:1] == '+' else i[2:] for i in dif 
                                                           if not i[:1] in '-?'])

def intelJoin(s1, s2):
    if s1 != '':
        if s2 == '':
            return s1
        else: 
            return s1 + ' and ' + s2
    else:
        return s2

## DFs
def mergeKeepLeftIndex(df1, df2):
    return df1.reset_index().merge(df2, how='left').set_index('index')

def swapCols(c1, c2, df):
    df = df.rename(columns={c1: 'tmp', c2: c1}); df = df.rename(columns={'tmp': c2})
    return df

def pasteCols(entry_col, score_col, df):
    return df[entry_col] + '(' + df[score_col].map(str) + ')'

## Plot Utils ===================================================
def loglog(series, xl, yl):
    """
    @brief      Count occurrence of values in the given series and 
                plot the resultant dist in scatter format. Suitable for power
                law dists.

    @param      series  a series of values having dups
    @param      xl      xlabel
    @param      yl      ylabel
    
    @return     scatter plot with log scales for both axis
    """
    h = series.value_counts()
    uniq_vals, counts = h.keys(), h.values
    
    fig = plt.figure()
    plt.scatter(x=uniq_vals, y=counts)
    plt.loglog()
    
    plt.xlabel(xl), plt.ylabel(yl)
    plt.xlim(min(uniq_vals), max(uniq_vals))
    plt.grid(True)
    return fig

# In each group, hide xticks of all subplots except the last one 
def hide_xticks(subplots, lasts):
    # hide xticks of all subplots
    plt.setp([a.get_xticklabels() for a in subplots.axes], visible=False)
    # turn on only ticks of last subplots in each group
    last_axes = [subplots.axes[i] for i in lasts]
    plt.setp([a.get_xticklabels() for a in last_axes], visible=True)

def setAxProps(ax, fontproperties, offset=False):
    ax.set_xticklabels(ax.get_xticks(), fontproperties)
    ax.set_yticklabels(ax.get_yticks(), fontproperties)
    
    # turn off decimal format on y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(offset)

# def setFontSize(ax, size):
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(size)

## Set funcs ===================================================
def unionAll(sets):
    res = set()
    for i in range(len(sets)):
        res = res.union(sets[i])

    return res

## File/Dir funcs
def rmExtension(fname):
    return fname.replace('.csv', '')

def listFiles(folder):
    fnames = os.listdir(folder)
    return map(rmExtension, fnames)

def mkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)