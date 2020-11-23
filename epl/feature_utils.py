import datetime as dt
import numpy as np
import pandas as pd

from epl.match_utils import full_table_calculator


def add_game_week(match_df):

    # create the columns init with nan
    match_df['HomeTeamGameWeek'] = np.nan
    match_df['AwayTeamGameWeek'] = np.nan
    # get col order to apply at end as combine_first changes col order
    df_cols = match_df.columns

    # ensure sorted by date, then create game week by season
    match_df = match_df.sort_values('Date', ascending=True)
    for s in match_df.Season.unique():
        # new df for only that season
        df_s = match_df[match_df.Season == s]
        # for each team assign game week (as already sorted by Date)
        for t in df_s.HomeTeam.unique():
            df_t = df_s[(df_s.HomeTeam == t) | (df_s.AwayTeam == t)]
            df_t['HomeTeamGameWeek'] = [x for x in range(1, len(df_t)+1)]
            df_t['AwayTeamGameWeek'] = [x for x in range(1, len(df_t)+1)]

            # where team is the HomeTeam, set GameWeek else nan, same for away
            df_t['HomeTeamGameWeek'] = np.where(
                df_t.HomeTeam == t, df_t['HomeTeamGameWeek'], np.nan)
            df_t['AwayTeamGameWeek'] = np.where(
                df_t.AwayTeam == t, df_t['AwayTeamGameWeek'], np.nan)

            # combine first onto the original df
            match_df = match_df.combine_first(
                df_t[['HomeTeamGameWeek', 'AwayTeamGameWeek']])

    # keep original col order and ensure date sorted again
    match_df = match_df[df_cols].sort_values(['Date'])
    return match_df


def add_league_pos(match_df):

    # first we need to use the match result data to compute the league tables at each point in time
    # 'point in time' is defined as the max date per game week

    # list to hold all the league tables at each week for each season
    league_tables = []

    # compute for each season
    for s in match_df.Season.unique():
        print("Computing league table for season: {}".format(s))
        # get only matches for that season and get the min game week per match up
        df_s = match_df[match_df.Season == s]
        df_s['MinGameWeek'] = df_s[[
            'HomeTeamGameWeek', 'AwayTeamGameWeek']].min(axis=1)

        # now we get the dates that correspond to each game week
        week_dates = df_s[['Date', 'MinGameWeek']].groupby(
            'MinGameWeek').max().to_dict()['Date']

        # for each week_date we then get the table as of that point in time
        for week, date in week_dates.items():
            # compute league table up to that point
            tab = full_table_calculator(df_s[df_s.Date <= date])
            # add on some other columns for the join
            tab['Season'] = s
            tab['GameWeek'] = week
            league_tables.append(tab)

    # concat into 1 big df
    tables_df = pd.concat(league_tables)
    # we want to only know table pre game so we need to add 1 pre merge join
    tables_df['GameWeek'] = tables_df['GameWeek'] + 1

    # create home and away versions to join on
    tables_df = tables_df.reset_index()
    tables_home_df = tables_df[['Season', 'Team', 'Points', 'LeagPos', 'GameWeek']].rename(
        columns={'Team': 'HomeTeam', 'GameWeek': 'HomeTeamGameWeek', 'Points': 'HomeTeamPoints', 'LeagPos': 'HomeLeagPos'})
    tables_away_df = tables_df[['Season', 'Team', 'Points', 'LeagPos', 'GameWeek']].rename(
        columns={'Team': 'AwayTeam', 'GameWeek': 'AwayTeamGameWeek', 'Points': 'AwayTeamPoints', 'LeagPos': 'AwayLeagPos'})

    # join them on
    match_df = pd.merge(left=match_df, right=tables_home_df, how='left', on=[
                        'Season', 'HomeTeamGameWeek', 'HomeTeam'])
    match_df = pd.merge(left=match_df, right=tables_away_df, how='left', on=[
                        'Season', 'AwayTeamGameWeek', 'AwayTeam'])

    return match_df, tables_df


def add_prev_season_pos(match_df, tables_df, promotion_treatment='20'):

    # create map of season to next season
    season_dict = dict(zip(tables_df.Season.unique(),
                           tables_df.Season.unique()[1:]))
    # get the last game week per season so we only get the end of season tables
    final_tables_df = tables_df[['Season', 'GameWeek']].groupby(
        'Season').max().reset_index()
    # create col for next season mapped from Season
    final_tables_df['NextSeason'] = final_tables_df['Season'].map(season_dict)

    # join onto original tables_df
    new_tables_df = pd.merge(
        left=tables_df, right=final_tables_df, how='left', on=['Season', 'GameWeek'])
    # if not present then not last game week so ditch
    final_tables_df = new_tables_df[~new_tables_df.NextSeason.isna()]
    # rename cols
    final_tables_df = final_tables_df[['Team', 'NextSeason', 'LeagPos']].rename(
        columns={'NextSeason': 'Season', 'LeagPos': 'PrevLeagPos'})

    # merge prev pos onto home and away cols of orig df
    match_df = pd.merge(left=match_df, right=final_tables_df.rename(columns={
                        'Team': 'HomeTeam', 'PrevLeagPos': 'HomePrevLeagPos'}), how='left', on=['Season', 'HomeTeam'])
    match_df = pd.merge(left=match_df, right=final_tables_df.rename(columns={
                        'Team': 'AwayTeam', 'PrevLeagPos': 'AwayPrevLeagPos'}), how='left', on=['Season', 'AwayTeam'])

    # drop first season as no prev season for it
    first_season = list(season_dict.keys())[0]
    df_final = match_df[match_df.Season != first_season]
    if promotion_treatment == '20':
        # fill blanks with P for promoted (as they didn't exist in prev year)
        df_final['HomePrevLeagPos'] = df_final['HomePrevLeagPos'].fillna(20)
        df_final['AwayPrevLeagPos'] = df_final['AwayPrevLeagPos'].fillna(20)
    else:
        # fill blanks with P for promoted (as they didn't exist in prev year)
        df_final['HomePrevLeagPos'] = df_final['HomePrevLeagPos'].fillna('P')
        df_final['AwayPrevLeagPos'] = df_final['AwayPrevLeagPos'].fillna('P')

    return df_final


def add_avg_ppg(match_df, streak_length):

    sl = str(streak_length)
    orig_cols = list(match_df.columns)

    h_map = {'H': 3, 'D': 1, 'A': 0}
    a_map = {'H': 0, 'D': 1, 'A': 3}

    # iterate over all the teams in the match_df (every team will play a home game so just just that)
    for team in df.HomeTeam.unique():
        # get the games for that team
        team_df = df[(df.HomeTeam == team) | (df.AwayTeam == team)]
        # now need to compute points for that team per game
        team_df['PPG'] = 0
        team_df.PPG = np.where(team_df.HomeTeam == team,
                               team_df.FTR.map(h_map), team_df.PPG)
        team_df.PPG = np.where(team_df.AwayTeam == team,
                               team_df.FTR.map(a_map), team_df.PPG)
        # compute mean, offset by 1 so we don't include result of current game
        team_df['AvgPPG_' +
                sl] = team_df.PPG.shift(1).rolling(streak_length).mean().fillna(0)
        # add new cols for Home and Away versions
        team_df['HomeAvgPPG_' +
                sl] = np.where(team_df.HomeTeam == team, team_df['AvgPPG_'+sl], np.nan)
        team_df['AwayAvgPPG_' +
                sl] = np.where(team_df.AwayTeam == team, team_df['AvgPPG_'+sl], np.nan)

        cols_to_keep = ['HomeAvgPPG_'+sl, 'AwayAvgPPG_'+sl]
        team_df = team_df[cols_to_keep]

        match_df = match_df.combine_first(team_df)

    return match_df[orig_cols + cols_to_keep]


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


if __name__ == '__main__':
    None
