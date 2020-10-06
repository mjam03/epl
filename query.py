import datetime as dt
import numpy as np
import os
import pandas as pd
import sqlite3


# data path for the sqlite database
DB_PATH = '/Users/jamisonm/dev/epl/data/match_results.sqlite'


def query_db(query):
    '''
    Generic function to hit the database and return results
    Query: str sql query statement
    '''
    try:
        conn = sqlite3.connect(DB_PATH)
    except:
        return "Unable to establish connection to {}".format(DB_PATH)
    
    try:
        print('Running query: {}'.format(query))
        res = pd.read_sql(query, conn)
    except:
        return "Unable to run query: {}".format(query)
    return res


def query_creator(table, cols=None, wc=None):
    '''
    Fn to format a query string using args
    table: str for the table in the db to query from
    cols: list of the columns we wish to query ['HomeTeam', 'AwayTeam']
    wc: dict , wc=
    '''
    if wc:
        conds = []
        for col,cond in wc.items():
            if isinstance(cond[1], str):
                conds.append("{} {} '{}'".format(col, cond[0], cond[1]))
            else:
                conds.append("{} {} {}".format(col, cond[0], cond[1]))
        wc = ' AND '.join(conds)
    else:
        wc = ''
    
    if cols:
        col_query = ', '.join(cols)
    else:
        col_query = '*'
    
    query = 'SELECT {} FROM {} WHERE {}'.format(col_query, table, wc)
    return query


def create_and_query(table, cols=None, wc=None):
    
    query = query_creator(table, cols, wc)
    res = query_db(query)
    return res


if __name__ == "__main__":
    None
 