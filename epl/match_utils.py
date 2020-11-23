import datetime as dt
import numpy as np
import pandas as pd
import sqlite3

from epl.query import create_and_query, create_conn


def result_calculator(match_results, res_type):
    """
    Function to output the league table for a set of matches including MP, GF, GA and points
    match_results: dataframe of match results including home and away team cols and score lines
    res_type: string of 'H' or 'A' - computes the results from a certain perspective
    """
    # check if we have the columns we need
    req_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    for col in req_cols:
        if col not in match_results.columns:
            return 'Missing column: {}, need following cols: {}'.format(col, req_cols)
    # handle whether perspective of H or A
    if res_type == 'H':
        # make everything from H perspective
        match_results = match_results.rename(
            columns={'HomeTeam': 'Team', 'AwayTeam': 'Opp', 'FTHG': 'GF', 'FTAG': 'GA'})
        # compute points from H perspective
        home_p = {'H': 3, 'A': 0, 'D': 1}
        home_res = {'H': 'W', 'A': 'L', 'D': 'D'}

        match_results['Points'] = match_results['FTR'].map(home_p)
        match_results['FTR'] = match_results['FTR'].map(home_res)

    elif res_type == 'A':
        # make everything from A perspective
        match_results = match_results.rename(
            columns={'AwayTeam': 'Team', 'HomeTeam': 'Opp', 'FTHG': 'GA', 'FTAG': 'GF'})
        # compute points from A perspective
        away_p = {'A': 3, 'H': 0, 'D': 1}
        away_res = {'A': 'W', 'H': 'L', 'D': 'D'}

        match_results['Points'] = match_results['FTR'].map(away_p)
        match_results['FTR'] = match_results['FTR'].map(away_res)
    else:
        return 'res_type must either be H or A, not: {}'.format(res_type)

    return match_results


def table_calculator(match_results, res_type):

    # work out from perspective we care about
    df_match = result_calculator(match_results, res_type)

    # agg by team and result
    df_match = df_match.groupby(['Team', 'FTR']).agg(
        {'Opp': 'count', 'GF': 'sum', 'GA': 'sum', 'Points': 'sum'}).reset_index()
    df_match = df_match.rename(columns={'Opp': 'MP'})

    # pivot by W/L/D
    df_res = pd.pivot_table(data=df_match[[
                            'Team', 'FTR', 'MP']], index='Team', columns='FTR', values='MP').fillna(0)

    # if there are no results for a given type, then create col with zero to complete table
    for res in ['W', 'D', 'L']:
        if res not in df_res.columns:
            df_res[res] = 0

    df_res_goals = df_match.groupby('Team').sum()
    df_res_goals['GD'] = df_res_goals['GF'] - df_res_goals['GA']
    df_res = pd.merge(left=df_res_goals, right=df_res, how='left',
                      on='Team').sort_values('Points', ascending=False)
    df_res['Loc'] = res_type

    df_res = df_res[['MP', 'W', 'L', 'D', 'GF', 'GA', 'GD', 'Points']]

    return df_res


def full_table_calculator(match_results):

    df_home = table_calculator(match_results, 'H')
    df_away = table_calculator(match_results, 'A')

    df_res = pd.concat([df_home, df_away])
    df_res = df_res.groupby('Team').sum().sort_values(
        ['Points', 'GD', 'GF'], ascending=False)
    df_res = df_res.reset_index().reset_index().rename(
        columns={'index': 'LeagPos'}).set_index('Team')
    df_res['LeagPos'] = df_res['LeagPos'] + 1
    df_res = df_res[[x for x in df_res.columns if x !=
                     'LeagPos'] + ['LeagPos']]

    return df_res


def league_table_asof(div, season, asof_date, conn=None):

    if not conn:
        conn = create_conn()

    # variable checking and error messages
    if not isinstance(div, str):
        try:
            elig_divs = pd.read_sql(
                """SELECT DISTINCT Div from matches""", conn)
            elig_divs = elig_divs['Div'].values
        except:
            return 'Cannot connect to db'
        conn.close()
        return "Div: {} not in db, must be from: {}".format(div, ", ".join(elig_divs))

    if not isinstance(season, str):
        try:
            elig_seasons = pd.read_sql(
                """SELECT DISTINCT Season from matches""", conn)
            elig_seasons = elig_seasons['Season'].values
        except:
            return 'Cannot connect to db'
        conn.close()
        return "Season: {} not in db, must be from: {}".format(season, ", ".join(elig_seasons))

    if not asof_date:
        asof_date = dt.date.today() + dt.timedelta(days=365*10)
        asof_date = pd.to_datetime(asof_date)
    else:
        if not isinstance(asof_date, pd.Timestamp):
            try:
                asof_date = pd.to_datetime(asof_date)
            except:
                return "Failed to convert asof_date: {} to datetime using pd.to_datetime".format(asof_date)

    # query required data from db
    table_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'] + ['Date']
    df_raw = create_and_query('matches', cols=table_cols, wc={
                              'Div': ['=', div], 'Season': ['=', season]})
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    df = df_raw[df_raw.Date <= asof_date]
    df_res = full_table_calculator(df)

    return df_res


def find_matches_by_score(score, is_ht=False, div=None, home_team=None, away_team=None, leading_team=None, losing_team=None):

    # form the where statement in the sql query as sql 10x-50x faster at filtering than pandas
    wc = {}
    if div:
        wc['Div'] = ['=', div]
    if home_team:
        wc['HomeTeam'] = ['=', home_team]
    if away_team:
        wc['AwayTeam'] = ['=', away_team]
    if len(wc) == 0:
        wc = None

    # get cols we care about
    cols = ['Div', 'Date', 'Season', 'HomeTeam',
            'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']
    # query
    df = create_and_query('matches', cols, wc).dropna()

    home_goals = 'HTHG' if is_ht else 'FTHG'
    away_goals = 'HTAG' if is_ht else 'FTAG'

    # create tuple for score and select where matches
    df['Score'] = list(zip(df[home_goals], df[away_goals]))
    df = df[(df['Score'] == score) | (df['Score'] == score[::-1])]

    # if leading / trailing team specified then apply that filter
    # don't know how to do this in sql yet so easier in pandas for now post sql query
    if leading_team:
        if score[0] == score[1]:
            df = df[(df['HomeTeam'] == leading_team) |
                    (df['AwayTeam'] == leading_team)]
        else:
            df = df[((df['HomeTeam'] == leading_team) & (df[home_goals] > df[away_goals])) | (
                (df['AwayTeam'] == leading_team) & (df[home_goals] < df[away_goals]))]
    if losing_team:
        if score[0] == score[1]:
            df = df[(df['HomeTeam'] == losing_team) |
                    (df['AwayTeam'] == losing_team)]
        else:
            df = df[((df['HomeTeam'] == losing_team) & (df[home_goals] < df[away_goals])) | (
                (df['AwayTeam'] == leading_team) & (df[home_goals] > df[away_goals]))]

    return df


if __name__ == "__main__":
    # function to check if working
    # x = league_table_asof('E0', '2019/2020', None, conn=None)
    # print(x)
    None
