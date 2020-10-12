import datetime as dt
import numpy as np
import os
import pandas as pd
import pathlib
import sqlite3


# data path for the sqlite database
dir_name = str(pathlib.Path(__file__).parent.absolute())
dir_name = dir_name.split('/')[:-1] + ['data', 'match_results.sqlite']
DB_PATH = "/".join(dir_name)


def create_conn():

    conn = sqlite3.connect(DB_PATH)
    return conn


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
        for col, cond in wc.items():
            if isinstance(cond[1], str):
                conds.append("{} {} '{}'".format(col, cond[0], cond[1]))
            else:
                conds.append("{} {} {}".format(col, cond[0], cond[1]))
        wc = 'WHERE ' + ' AND '.join(conds)
    else:
        wc = ''

    if cols:
        col_query = ', '.join(cols)
    else:
        col_query = '*'

    query = 'SELECT {} FROM {} {}'.format(col_query, table, wc)
    return query


def create_and_query(table, cols=None, wc=None):

    query = query_creator(table, cols, wc)
    res = query_db(query)
    return res


if __name__ == "__main__":
    None
