import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from epl.match_utils import full_table_calculator

COL_MAP = {'GF': {'Home': 'FTHG', 'Away': 'FTAG'},
           'GA': {'Home': 'FTAG', 'Away': 'FTHG'},
           'SF': {'Home': 'HS', 'Away': 'AS'},
           'SA': {'Home': 'AS', 'Away': 'HS'},
           'STF': {'Home': 'HST', 'Away': 'AST'},
           'STA': {'Home': 'AST', 'Away': 'HST'},
           'PPG': {'Home': 'FTR', 'Away': 'FTR'}
           }


def add_game_week(match_df):

    # create the columns init with nan
    match_df['HGW'] = np.nan
    match_df['AGW'] = np.nan
    # get col order to apply at end as combine_first changes col order
    df_cols = match_df.columns

    # ensure sorted by date, then create game week by season
    match_df = match_df.sort_values('Date', ascending=True)
    for d in match_df.Div.unique():
        print('Adding game week for div: {}'.format(d))
        for s in match_df.Season.unique():
            # new df for only that season
            df_s = match_df[(match_df.Season == s) & (match_df.Div == d)]
            # for each team assign game week (as already sorted by Date)
            for t in df_s.HomeTeam.unique():
                df_t = df_s[(df_s.HomeTeam == t) | (df_s.AwayTeam == t)]
                df_t['GW'] = [x for x in range(1, len(df_t)+1)]

                # where team is the HomeTeam, set GameWeek else nan, same for away
                df_t['HGW'] = np.where(
                    df_t.HomeTeam == t, df_t['GW'], np.nan)
                df_t['AGW'] = np.where(
                    df_t.AwayTeam == t, df_t['GW'], np.nan)
                df_t = df_t.drop(columns=['GW'])

                # combine first onto the original df
                match_df = match_df.combine_first(
                    df_t[['HGW', 'AGW']])

    # keep original col order and ensure date sorted again
    match_df = match_df[df_cols].sort_values(['Date'])
    return match_df


def add_league_pos(match_df):

    # first we need to use the match result data to compute the league tables at each point in time
    # 'point in time' is defined as the max date per game week

    # list to hold all the league tables at each week for each season
    league_tables = []

    # for each div, season pair
    for d in match_df.Div.unique():
        for s in match_df[match_df.Div == d].Season.unique():
            print("Computing league table for div: {}, season: {}".format(d, s))
            # get relevant matches
            df = match_df[(match_df.Season == s) & (match_df.Div == d)]
            # get all game weeks
            game_weeks = df.HGW.unique()
            # for each game week
            for g in game_weeks:
                # calc the table
                tab = full_table_calculator(
                    df[(df.HGW <= g) | (df.AGW <= g)])
                # add on some other relevant data for join
                tab['Div'] = d
                tab['Season'] = s
                tab['GW'] = g
                league_tables.append(tab)

    # concat into 1 big df
    tables_df = pd.concat(league_tables)
    # we want to only know table pre game so we need to add 1 pre merge join
    tables_df['JoinGW'] = tables_df['GW'] + 1

    # create home and away versions to join on
    tables_df = tables_df.reset_index()
    tables_home_df = tables_df[['Div', 'Season', 'Team', 'Points', 'LeagPos', 'JoinGW']].rename(
        columns={'Team': 'HomeTeam', 'JoinGW': 'HGW', 'Points': 'HPoints', 'LeagPos': 'HLeagPos'})
    tables_away_df = tables_df[['Div', 'Season', 'Team', 'Points', 'LeagPos', 'JoinGW']].rename(
        columns={'Team': 'AwayTeam', 'JoinGW': 'AGW', 'Points': 'APoints', 'LeagPos': 'ALeagPos'})

    # join them on
    match_df = pd.merge(left=match_df, right=tables_home_df, how='left', on=[
                        'Div', 'Season', 'HGW', 'HomeTeam'])
    match_df = pd.merge(left=match_df, right=tables_away_df, how='left', on=[
                        'Div', 'Season', 'AGW', 'AwayTeam'])

    match_df['HPoints'] = match_df['HPoints'].fillna(0)
    match_df['APoints'] = match_df['APoints'].fillna(0)
    match_df['HLeagPos'] = np.where(
        match_df.HGW == 1, 10, match_df['HLeagPos'])
    match_df['ALeagPos'] = np.where(
        match_df.AGW == 1, 10, match_df['ALeagPos'])

    return match_df, tables_df


def add_prev_season_pos(match_df, tables_df, promotion_treatment='20'):

    # create map of season to next season
    season_dict = dict(zip(tables_df.Season.unique(),
                           tables_df.Season.unique()[1:]))
    # get the last game week per season so we only get the end of season tables
    final_tables_df = tables_df[['Div', 'Season', 'GW']].groupby(
        ['Div', 'Season']).max().reset_index()
    # create col for next season mapped from Season
    final_tables_df['NextSeason'] = final_tables_df['Season'].map(season_dict)

    # join onto original tables_df
    new_tables_df = pd.merge(
        left=tables_df, right=final_tables_df, how='left', on=['Div', 'Season', 'GW'])
    # if not present then not last game week so ditch
    final_tables_df = new_tables_df[~new_tables_df.NextSeason.isna()]
    # rename cols
    final_tables_df = final_tables_df[['Team', 'Div', 'NextSeason', 'LeagPos']].rename(
        columns={'NextSeason': 'Season', 'LeagPos': 'PrevLeagPos'})

    # merge prev pos onto home and away cols of orig df
    match_df = pd.merge(left=match_df, right=final_tables_df.rename(columns={
                        'Team': 'HomeTeam', 'PrevLeagPos': 'HPrevLeagPos'}), how='left', on=['Div', 'Season', 'HomeTeam'])
    match_df = pd.merge(left=match_df, right=final_tables_df.rename(columns={
                        'Team': 'AwayTeam', 'PrevLeagPos': 'APrevLeagPos'}), how='left', on=['Div', 'Season', 'AwayTeam'])

    # drop first season as no prev season for it
    first_season = list(season_dict.keys())[0]
    df_final = match_df[match_df.Season != first_season]
    if promotion_treatment == '20':
        # fill blanks with P for promoted (as they didn't exist in prev year)
        df_final['HPrevLeagPos'] = df_final['HPrevLeagPos'].fillna(20)
        df_final['APrevLeagPos'] = df_final['APrevLeagPos'].fillna(20)
    else:
        # fill blanks with P for promoted (as they didn't exist in prev year)
        df_final['HPrevLeagPos'] = df_final['HPrevLeagPos'].fillna('P')
        df_final['APrevLeagPos'] = df_final['APrevLeagPos'].fillna('P')

    return df_final


def add_avg_cols(match_df, cols, streak, avg_type='mean'):

    # str version of streak for col naming
    sl = str(streak)
    # get original cols as combine first changes col order
    orig_cols = list(match_df.columns)

    # point map
    h_map = {'H': 3, 'D': 1, 'A': 0}
    a_map = {'H': 0, 'D': 1, 'A': 3}

    # define only col map we need based on cols
    col_m = {k: v for k, v in COL_MAP.items() if k in cols}
    # get raw_cols needed
    raw_cols = [list(x.values()) for x in col_m.values()]
    raw_cols = list(set([j for i in raw_cols for j in i]))
    # issue error if not all raw columns are present based on demand
    for c in raw_cols:
        if c not in match_df.columns:
            prob_cols = [k for k, v in col_m.items() if 'FTHG' in v.values()]
            print('Unable to calc: {} without raw column: {}'.format(
                prob_cols.join(', '), c))

    # now we iterate over all the teams in the df and create the cols
    teams = list(set(list(match_df.HomeTeam.unique()) +
                     list(match_df.AwayTeam.unique())))
    for t in teams:
        # create list for final output cols to keep order
        out_cols = []
        # get only those teams
        t_df = match_df[(match_df.HomeTeam == t) | (
            match_df.AwayTeam == t)].sort_values('Date')

        # now iterate over the cols we want and create the new col
        for c, v in col_m.items():
            # initialise the new cols with value 0
            t_df[c] = 0
            # special point map treatment for PPG based on FTR
            if c == 'PPG':
                t_df[c] = np.where(t_df.HomeTeam == t, t_df[v['Home']].map(
                    h_map), t_df[v['Home']].map(a_map))
            else:
                # now add an un-averaged version based on whether home or away
                t_df[c] = np.where(t_df.HomeTeam == t,
                                   t_df[v['Home']], t_df[v['Away']])

            # now compute backward looking avg col value and shift by 1 (to not include current game)
            c_name = 'Avg'+c+'_'+sl
            if avg_type == 'mean':
                t_df[c_name] = t_df[c].rolling(streak).apply(np.mean).shift(1)
            elif avg_type == 'exp':
                c_name = 'Exp'+c_name
                t_df[c_name] = t_df[c].ewm(span=streak).apply(np.mean).shift(1)
            else:
                t_df[c_name] = t_df[c].rolling(streak).apply(avg_type).shift(1)

            # now we need to create 'Home' and 'Away' versions of cols for combine_first
            # populate col where not t with np.nan for combine_first
            h_col = 'H'+c_name
            a_col = 'A'+c_name
            out_cols.append(h_col)
            out_cols.append(a_col)
            t_df[h_col] = np.where(t_df.HomeTeam == t, t_df[c_name], np.nan)
            t_df[a_col] = np.where(t_df.AwayTeam == t, t_df[c_name], np.nan)

        # now we have our averaged df for the team t - we now combine it onto the original df
        t_df = t_df[out_cols]
        match_df = match_df.combine_first(t_df)

    # order the cols the way we want, fix date sorting, return
    match_df = match_df[orig_cols + out_cols].sort_values('Date')
    return match_df


def add_result_streak(match_df, streak_length):

    orig_cols = list(match_df.columns)
    for team in match_df.HomeTeam.unique():
        match_df = add_team_result_streak(match_df, streak_length, team)

    cols = ['HomeWStreak_', 'HomeDStreak_', 'HomeLStreak_',
            'AwayWStreak_', 'AwayDStreak_', 'AwayLStreak_']
    match_df = match_df[orig_cols + [x+str(streak_length) for x in cols]]
    return match_df


def add_team_result_streak(match_df, streak_length, team):

    match_df = match_df.sort_values('Date')
    # get only the matches that the team has played in
    team_df = match_df[(match_df['HomeTeam'] == team) |
                       (match_df['AwayTeam'] == team)]
    # store the original index - required to combine_first back on to match_df later
    orig_index = team_df.index

    # create new 'Team' and 'Opp' cols to help categorise W/D/L
    team_df['Team'] = team
    team_df['Opp'] = np.where(team_df['HomeTeam'] ==
                              team, team_df['AwayTeam'], team_df['HomeTeam'])

    # convert FTR H/A/D to W/D/L from perspective of team we care about
    h_map = {'H': 'W', 'D': 'D', 'A': 'L'}
    a_map = {'H': 'L', 'D': 'D', 'A': 'W'}
    team_df['Team_FTR'] = np.where(team_df['HomeTeam'] == team, team_df['FTR'].map(
        h_map), team_df['FTR'].map(a_map))

    # pivot W/D/L so we can start to create streak cols
    # we piv on the FTR col, count and fillna and then compute an offset by 1 streak (so we don't include current result)
    piv_df = pd.pivot_table(data=team_df, index=[
                            'Date', 'Team', 'HomeTeam', 'AwayTeam'], columns='Team_FTR', values='Opp', aggfunc='count')
    piv_df = piv_df.fillna(0).shift(1).rolling(
        streak_length).sum() / streak_length
    piv_df = piv_df.reset_index()

    # create the cols for Home vs Away and drop un-needed cols
    sl = str(streak_length)
    try:
        piv_df['HomeWStreak_' +
               sl] = np.where(piv_df['HomeTeam'] == team, piv_df['W'], np.nan)
        piv_df['HomeDStreak_' +
               sl] = np.where(piv_df['HomeTeam'] == team, piv_df['D'], np.nan)
        piv_df['HomeLStreak_' +
               sl] = np.where(piv_df['HomeTeam'] == team, piv_df['L'], np.nan)
        piv_df['AwayWStreak_' +
               sl] = np.where(piv_df['AwayTeam'] == team, piv_df['W'], np.nan)
        piv_df['AwayDStreak_' +
               sl] = np.where(piv_df['AwayTeam'] == team, piv_df['D'], np.nan)
        piv_df['AwayLStreak_' +
               sl] = np.where(piv_df['AwayTeam'] == team, piv_df['L'], np.nan)
        piv_df = piv_df.drop(
            columns=['Date', 'Team', 'HomeTeam', 'AwayTeam', 'W', 'D', 'L'])
        piv_df = piv_df.set_index(orig_index)

        match_df = match_df.combine_first(piv_df)
    except:
        print('Could not generate for team: {}'.format(team))

    return match_df


def home_away_to_team_opp(df):
    '''
    Accepts df of n matches, converts to 2n rows
    HomeTeam / Away Team converted to Team / Opp
    '''
    df_home = df.copy()
    df_home = df_home.rename(columns={'HomeTeam': 'Team', 'AwayTeam': 'Opp'})
    df_home['Home'] = 1

    df_away = df.copy()
    df_away = df_away.rename(columns={'AwayTeam': 'Team', 'HomeTeam': 'Opp'})
    df_away['Home'] = 0

    df_m = pd.concat([df_home, df_away]).sort_values(['Date', 'Team'])
    df_m['GF'] = np.where(df_m['Home'] == 1, df_m['FTHG'], df_m['FTAG'])
    df_m['GA'] = np.where(df_m['Home'] == 1, df_m['FTAG'], df_m['FTHG'])
    df_m = df_m.drop(columns=['FTHG', 'FTAG'])

    return df_m


def create_goal_probs(lambdas, max_goals):
    '''
    Returns an array of arrays for goal probabilities
    Accepts an array of poisson lambdas
    '''
    # form goal array from 0 to max_goals
    goal_array = np.arange(0, max_goals + 1)

    if isinstance(lambdas, pd.Series):
        # convert to np array as much faster
        lambdas = lambdas.values
        goal_probs = [poisson.pmf(goal_array, x) for x in lambdas]
    elif isinstance(lambdas, np.ndarray):
        goal_probs = [poisson.pmf(goal_array, x) for x in lambdas]
    return goal_probs


def create_poisson_prediction_output(eval_df, df, other_data_cols):

    eval_df['GoalProbs'] = create_goal_probs(eval_df['lambda'], 5)
    # get the most likely score from the probability array
    # np 2-3x faster than pandas apply here
    # e.g. vs .apply(lambda x: np.argmax(x) + 1)
    eval_df['MaxProbGoals'] = np.argmax(
        np.stack(eval_df['GoalProbs'].values), axis=1)+1

    # get key cols to reformat into per match data
    match_id_cols = ['Date', 'Team', 'Opp']
    eval_df = pd.merge(
        left=eval_df, right=df[match_id_cols], how='left', left_index=True, right_index=True)

    eval_df = eval_df_to_match_eval_df(
        eval_df, df, match_id_cols, other_data_cols)
    eval_df = create_match_prediction_stats(eval_df)

    return eval_df


def eval_df_to_match_eval_df(eval_df, df, match_id_cols, other_cols):

    match_result = ['FTR']

    home_eval = eval_df[eval_df.Home == 1]
    # join other data on here to minimise joins
    home_eval = pd.merge(
        home_eval, right=df[match_id_cols + other_cols + match_result], how='left', on=match_id_cols)

    home_eval = home_eval.rename(columns={'GF': 'FTHG', 'Team': 'HomeTeam', 'Opp': 'AwayTeam',
                                          'GoalProbs': 'HomeGoalProbs', 'MaxProbGoals': 'HomeMaxProbGoals', 'lambda': 'HomeLambda'})
    home_eval = home_eval.drop(columns=['Home'])

    away_eval = eval_df[eval_df.Home == 0]
    away_eval = away_eval[['Date', 'Team', 'GF',
                           'lambda', 'GoalProbs', 'MaxProbGoals']]
    away_eval = away_eval.rename(columns={'GF': 'FTAG', 'Team': 'AwayTeam',
                                          'GoalProbs': 'AwayGoalProbs', 'MaxProbGoals': 'AwayMaxProbGoals', 'lambda': 'AwayLambda'})

    # model trained on indiv teams, not matches
    # this means for a given match, not necessarily both teams in the training data set
    # we thus do an inner join and drop some data here (where only 1 team in match in training data)
    df_eval = pd.merge(left=home_eval, right=away_eval,
                       how='inner', on=['Date', 'AwayTeam'])

    # reorder cols
    key_cols = ['Date', 'HomeTeam', 'AwayTeam']
    goal_cols = ['FTHG', 'FTAG']
    model_cols = [y+x for x in ['Lambda', 'GoalProbs', 'MaxProbGoals']
                  for y in ['Home', 'Away']]
    df_eval = df_eval[key_cols + other_cols +
                      match_result + goal_cols + model_cols]
    return df_eval


def create_match_prediction_stats(df_eval):

    df_eval['GoalMatrix'] = [np.outer(x, y) for x, y in zip(
        df_eval.HomeGoalProbs.values, df_eval.AwayGoalProbs.values)]

    df_eval['AwayProb'] = [np.triu(x, 1).sum()
                           for x in df_eval['GoalMatrix'].values]
    df_eval['DrawProb'] = [np.trace(x).sum()
                           for x in df_eval['GoalMatrix'].values]
    df_eval['HomeProb'] = [np.tril(x, -1).sum()
                           for x in df_eval['GoalMatrix'].values]
    df_eval['FTRProbs'] = list(
        zip(df_eval['AwayProb'], df_eval['DrawProb'], df_eval['HomeProb']))

    res = ['A', 'D', 'H']
    df_eval['FTRPred'] = df_eval['FTRProbs'].apply(
        lambda x: res[x.index(max(x))])
    df_eval['Score'] = list(zip(df_eval['FTHG'], df_eval['FTAG']))
    df_eval['MaxProbScore'] = list(
        zip(df_eval['HomeMaxProbGoals'], df_eval['AwayMaxProbGoals']))
    return df_eval


if __name__ == '__main__':
    None
