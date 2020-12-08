import datetime as dt
import numpy as np
import os
import pandas as pd
import pathlib
import sqlite3


# define the site root
SITE_ROOT = 'https://www.football-data.co.uk/'
DATA_ROOT = 'https://www.football-data.co.uk/data.php'
DB_NAME = 'footie.sqlite'
DB_NAME_UAT = 'footie_uat.sqlite'


def get_root_dir():

    script_location = os.path.realpath(__file__)
    root_dir = script_location.split('/')[:-2]
    return '/'.join(root_dir)


def create_conn(uat=False):
    if uat:
        db_path = '/'.join([get_root_dir(), 'data', DB_NAME_UAT])
    else:
        db_path = '/'.join([get_root_dir(), 'data', DB_NAME])
    conn = sqlite3.connect(db_path)
    return conn


def table_exists(table_name, uat=False):

    try:
        res = query_db("SELECT name FROM sqlite_master WHERE type='table' AND name='{}'".format(
            table_name), uat=uat)
        if len(res) > 0:
            return True
        else:
            return False
    except:
        print("Unable to query db - likely whole db doesn't exist")
        return False


def get_table_columns(table_name, uat=False):
    '''
    Returns list of column names for table
    '''
    # establish connection
    conn = create_conn(uat=uat)
    # query and get cols
    if table_exists(table_name, uat=uat):
        cursor = conn.execute('SELECT * from {} '.format(table_name))
        cols = [x[0]for x in cursor.description]
    else:
        print("Table {} doesn't exist - returning empty cols".format(table_name))
        cols = []
    return cols


def query_db(query, uat=False):
    '''
    Generic function to hit the database and return results
    Query: str sql query statement
    '''
    try:
        conn = create_conn(uat=uat)
    except:
        return "Unable to establish connection to {}".format(DB_NAME)

    try:
        print('Running query: {}'.format(query))
        res = pd.read_sql(query, conn)
        for d in ['Date', 'AsOfDate']:
            if d in res.columns:
                res[d] = pd.to_datetime(res[d])

        # sort by cols if they are available so by def we get cols in chronological / grouped order
        sort_cols = ['AsOfDate', 'Date', 'Country', 'Div']
        sort_cols = [x for x in sort_cols if x in res.columns]
        if len(sort_cols) > 0:
            res = res.sort_values(sort_cols)
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
        cols = ['[{}]'.format(x) if (
            x in ['AS', 'FROM', 'WHERE', 'SELECT']) else x for x in cols]
        col_query = ', '.join(cols)
    else:
        col_query = '*'

    query = 'SELECT {} FROM {} {}'.format(col_query, table, wc)
    return query


def create_and_query(table, uat=False, cols=None, wc=None):

    query = query_creator(table, cols, wc)
    res = query_db(query, uat=uat)
    for d in ['Date', 'AsOfDate']:
        if d in res.columns:
            res[d] = pd.to_datetime(res[d])
    return res


if __name__ == "__main__":
    None
