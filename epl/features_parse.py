import datetime as dt
from functools import reduce
import numpy as np
import pandas as pd

from epl.query import create_and_query, create_conn, get_table_columns, query_creator, query_db, table_exists

FEATURE_KEY_COLS = ['Date', 'Team']
FEATURE_ID_COLS = ['Country', 'Div', 'Season']


def create_features_key_col(df):
    '''
    Returns orig df with a new key column
    Key is concat of str() date with team name
    '''
    if 'Date' in df.columns and 'Team' in df.columns:
        df['Key'] = df['Date'].apply(
            lambda x: x.strftime('%Y-%m-%d')+'_') + df['Team']
        return df
    else:
        print('Date and Team must be present in the df columns')
        return None


def get_current_feature_keys(table_name, uat=False):
    '''
    Returns list of keys currently in the features table
    Used to calc what new matches need features computed
    '''
    if table_exists(table_name, uat=uat):
        try:
            df = create_and_query(table_name, uat=uat, cols=FEATURE_KEY_COLS)
            df = create_features_key_col(df)
            curr_keys = list(df.Key.values)
            return curr_keys
        except:
            print('Failed trying to query {} table'.format(table_name))
            return []
    else:
        print("{} table doesn't exist".format(table_name))
        return []


def get_new_matches(fixtures=False, uat=False):
    '''
    Returns df of [Date, Team] for matches with no matching feature data
    Gets current feature and match keys, diffs and returns matches
    '''
    # first query feature table to get the key cols
    if fixtures:
        curr_feat_keys = get_current_feature_keys('fixtures_features', uat=uat)
    else:
        curr_feat_keys = get_current_feature_keys('features', uat=uat)

    # next query matches table for the key cols for comparison
    desired_cols = ['Date', 'HomeTeam', 'AwayTeam'] + FEATURE_ID_COLS
    if fixtures:
        curr_match_keys = create_and_query(
            'fixtures', uat=uat, cols=desired_cols)
        curr_match_keys = curr_match_keys.drop_duplicates()
    else:
        curr_match_keys = create_and_query(
            'matches', uat=uat, cols=desired_cols)

    # transform HomeTeam and AwayTeam into singular Team column
    curr_match_keys = convert_home_away_df(curr_match_keys)
    curr_match_keys = create_features_key_col(curr_match_keys)

    new_matches = curr_match_keys[~curr_match_keys.Key.isin(curr_feat_keys)]
    new_matches = new_matches.drop(columns=['Key'])
    return new_matches


def get_feat_col_names(feats, streak_length, avg_type):
    '''
    Returns a dict of {[feat col name]: [feat_col_base]}
    e.g. for 'GF' feat, sl = 3, avg_type='Avg'
    output would be {AvgGF_3: GF}
    '''
    feat_cols = [avg_type+x+'_'+str(streak_length) for x in feats]
    return dict(zip(feat_cols, feats))


def get_feat_required_cols(feats):
    '''
    Returns dict of {feat_base: req_cols}
    '''
    return {k: list(v.values()) for k, v in feats.items()}


def create_col_map(feats, streak_length, avg_type):
    '''
    Returns dict of feat_col_name to component data
    e.g. {'AvgGF_3': {'GF': {'Home': 'FTHG', 'Away': 'FTAG'}}}
    '''
    col_map = {k: {v: feats[v]} for k, v in get_feat_col_names(
        feats.keys(), streak_length, avg_type).items()}
    return col_map


def convert_home_away_df(df):
    '''
    Returns df with HomeTeam / AwayTeam cols melted into {Team, Location}
    Required as feature data stored as {Date,Team}
    for ease of creation and time series analysis
    '''
    col_map = {'HomeTeam': 'Home', 'AwayTeam': 'Away'}
    other_cols = [x for x in df.columns if x not in col_map.keys()]

    df = pd.melt(df, id_vars=other_cols, var_name='Location',
                 value_vars=col_map.keys(), value_name='Team')
    df['Location'] = df['Location'].map(col_map)

    new_key = ['Date', 'Team', 'Location']
    new_cols = new_key + [x for x in df.columns if x not in new_key]
    df = df.sort_values(new_key)
    df = df[new_cols]
    return df


def create_base_feat_cols(df, fts, loc):
    '''
    Returns df with base cols required added on
    df: df of raw match data with required raw cols e.g. FTHG, FTAG
    fts: dict of feature col_map
    loc: str enum of ['All', 'Home', 'Away']
    '''
    df_new = df.copy()
    for k, v in fts.items():
        # get building blocks
        ft = list(v.values())[0]
        col_name = list(v.keys())[0]

        finish_a_or_h = (('A' == col_name[-1:]) or ('H' == col_name[-1:]))
        if finish_a_or_h and (col_name[:-1] in df_new.columns):
            df_new[col_name] = df_new[col_name[:-1]]
        else:
            if 'PPG' in col_name:
                # need to map to points
                h_map = {'H': 3, 'D': 1, 'A': 0}
                a_map = {'H': 0, 'D': 1, 'A': 3}
                if loc == 'All':
                    df_new[col_name] = np.where(df.Location == 'Home', df_new[ft['Home']].map(
                        h_map), df_new[ft['Away']].map(a_map))
                elif loc == 'Home':
                    df_new[col_name] = df_new[ft['Home']].map(h_map)
                elif loc == 'Away':
                    df_new[col_name] = df_new[ft['Away']].map(a_map)
            else:
                if loc == 'All':
                    df_new[col_name] = np.where(
                        df_new.Location == 'Home', df_new[ft['Home']], df_new[ft['Away']])
                elif loc == 'Home':
                    df_new[col_name] = df_new[ft['Home']]
                elif loc == 'Away':
                    df_new[col_name] = df_new[ft['Away']]

    return df_new


def split_home_away_feats(feats):
    '''
    Splits feature col map into 3 consti dicts
    Required to then create relevant feats separately
    '''
    # need to split out
    all_feats = {}
    home_feats = {}
    away_feats = {}

    for k, v in feats.items():

        const_cols = list(v.values())[0]
        home_away = const_cols.keys()
        if 'Home' in home_away and 'Away' in home_away:
            all_feats[k] = v
        elif 'Away' not in home_away:
            home_feats[k] = v
        elif 'Home'not in home_away:
            away_feats[k] = v

    return all_feats, home_feats, away_feats


def get_feats_raw_data(feats, uat=False):
    '''
    Returns raw matches df used for feature calculation
    Gets the required key cols as well as extra ID cols
    '''

    # get the required raw data columns from the required feats dict
    req_cols_dict = get_feat_required_cols(feats)
    req_cols = list(set([j for i in req_cols_dict.values() for j in i]))

    # define query
    key_cols = ['Date', 'HomeTeam', 'AwayTeam']
    all_cols = key_cols + req_cols
    if table_exists('matches', uat=uat):
        df = create_and_query('matches', uat=uat, cols=all_cols)
        df = convert_home_away_df(df)
        df = df.sort_values(FEATURE_KEY_COLS[::-1])
        return df
    else:
        print("matches table doesn't exist - need to create it first")
        return None


def calc_rolling_avg(df, feats, streak_length):
    '''
    Accept df of raw data, col_map dict of feats and avg length
    Computes rolling avg data for the specified features and renames cols
    '''

    # get the cols to avg then rename
    col_to_avg = [list(x.keys())[0] for x in feats.values()]
    col_rename_dict = dict(zip(col_to_avg, feats.keys()))

    # get the key cols and create new df with only data we need
    key_cols = FEATURE_KEY_COLS + ['Location']
    # df_ft forms our basis to which we will lj on the avg features
    df_ft = df[key_cols + col_to_avg]

    # create mean, shift and remove incorrectly shifted values
    df_feats = df_ft[col_to_avg].groupby(
        df_ft['Team']).rolling(streak_length).mean()

    # rename cols and reset index to only orig index (not Team included)
    df_feats = df_feats.rename(columns=col_rename_dict)
    df_feats = df_feats.reset_index()
    df_feats = df_feats.set_index('level_1')

    # lj onto the original data, sort and return
    feat_cols = col_rename_dict.values()
    df_out = pd.merge(
        left=df_ft, right=df_feats[feat_cols], how='left', left_index=True, right_index=True)
    df_out = df_out.sort_values(FEATURE_KEY_COLS[::-1])
    return df_out


def merge_home_away(ft_dfs, all_feats, home_feats, away_feats, shift=True):

    # join them and handle conditional ffill
    key_cols = FEATURE_KEY_COLS + ['Location']
    # order list of dfs so 'All' is first
    ft_d = [ft_dfs['All'], ft_dfs['Home'], ft_dfs['Away']]
    df_merged = reduce(lambda left, right: pd.merge(
        left, right, on=key_cols, how='left'), ft_d)

    # now we want to ffill all home and away cols
    # NaNs created when joining home / away on to all match data
    # as e.g. AvgGFA_3 doesn't exist on rows where Location == 'Home'
    # we thus need to ffill this PER TEAM from the previous Away result

    # create list of home and away specific feature cols
    home_away_cols = list(home_feats.keys()) + list(away_feats.keys())
    # ffill these cols once grouped by team
    df_merged[home_away_cols] = df_merged[['Team'] +
                                          home_away_cols].groupby(['Team']).ffill()

    if shift:
        # now we need to shift games down 1
        # this is to prevent taking into account the score in todays game
        # i.e. all avg features should be backward looking (and not include obs today)
        # e.g. 0-0 draw in last 3 games but today match 6-0
        # AvgGF_3 should be 0, not (6+0+0)/3 = 2
        # however we only do this for home and away feats for those games
        # the above fwd fill procedure handles away feats for home matches and vice versa
        df_merged.update(df_merged[all_feats.keys()].groupby(
            df_merged['Team']).shift(1))
        df_merged.update(df_merged[df_merged.Location == 'Home'][home_feats.keys()].groupby(
            df_merged['Team']).shift(1))
        df_merged.update(df_merged[df_merged.Location == 'Away'][away_feats.keys()].groupby(
            df_merged['Team']).shift(1))

    return df_merged


def handle_feats(feat_list, fixtures=False):

    # for each list of features, applies correct function to args
    # concats them all together columnwise at the end
    feat_dfs = []

    # get the new matches we need to compute for
    df_new_matches = get_new_matches(fixtures=fixtures)
    if len(df_new_matches) == 0:
        print('No new matches to process features for')
        return None

    # now iterate over each feature set, get raw data, compute and return
    for feat_desc in feat_list:
        if feat_desc['feat_type'] == 'avg':
            # unpack args
            feats = feat_desc['feat_dict']
            streak_length = feat_desc['streak']
            avg_type = feat_desc['avg_type']

            # get the raw data required to calc the features
            df_raw = get_feats_raw_data(feats)
            # restrict to only teams in the new matches df
            df_raw = df_raw[df_raw.Team.isin(df_new_matches.Team.unique())]

            # create map from feat_name to base_name to construction
            # split out by All / Home / Away for sequential calc
            col_map = create_col_map(feats, streak_length, avg_type)
            all_feats, home_feats, away_feats = split_home_away_feats(col_map)

            # create base cols e.g. GF / GA for use to calc e.g. AvgGF_3
            c_dict = {'All': all_feats, 'Home': home_feats, 'Away': away_feats}
            for k, v in c_dict.items():
                if len(v) > 0:
                    df_raw = create_base_feat_cols(df_raw, v, k)
                elif k == 'All':
                    # if all feats blank then issue so report
                    print('All features is blank - probably an error: {}'.format(v))

            # compute feats
            ft_dfs = {}
            for k, v in c_dict.items():
                # if not all, then restrict data to only home/away games
                if k != 'All':
                    df_r = df_raw[df_raw.Location == k]
                    df_f = calc_rolling_avg(df_r, v, streak_length)
                else:
                    df_f = calc_rolling_avg(df_raw, v, streak_length)
                # add to dict
                ft_dfs[k] = df_f

            df_feats = merge_home_away(
                ft_dfs, all_feats, home_feats, away_feats, shift=not(fixtures))

            # now we have our correctly offset feats
            # we need to select just the cols we need
            id_cols = FEATURE_KEY_COLS + ['Location']
            df_feats = df_feats[id_cols + list(col_map.keys())]

            feat_dfs.append(df_feats)
        else:
            print('Not supported yet')

    # now we concat them altogether along common key
    key_cols = FEATURE_KEY_COLS + ['Location']
    df_merged = reduce(lambda left, right: pd.merge(
        left, right, on=key_cols, how='outer'), feat_dfs)

    # now we only return the matches we needed
    if fixtures:
        # drop location for fixtures as we want most recent game regardless
        df_merged = df_merged.drop(columns=['Location'])
        # sort by date, then team for asof join
        df_merged = df_merged.sort_values(FEATURE_KEY_COLS)
        # backwards as of join the data on
        df_final = pd.merge_asof(df_new_matches, df_merged, on='Date', by=[
                                 'Team'], direction='backward', allow_exact_matches=False)
    else:
        df_final = pd.merge(left=df_new_matches,
                            right=df_merged, how='left', on=key_cols)
    return df_final


def get_requested_feat_cols(feat_list):

    feats = []
    for f in feat_list:
        fts = f['feat_dict']
        sl = f['streak']
        avg_type = f['avg_type']
        ft_col_map = create_col_map(fts, sl, avg_type)
        for k in ft_col_map.keys():
            feats.append(k)

    return feats


def process_feature_data(feat_list, fixtures=False, uat=False):
    '''
    Handles process of:
     - Identifying new matches that require features
     - Checking if the features requested line up to the columns in the feature table already
     - Compute new features data
     - Set/append down into sqlite
    '''
    if fixtures:
        table_name = 'fixtures_features'
    else:
        table_name = 'features'
    # first check if the columns requested are equal to those in the table
    # will handle this in future but for now just throw an error
    req_feats = get_requested_feat_cols(feat_list)
    new_feat_cols = FEATURE_KEY_COLS + \
        ['Location'] + FEATURE_ID_COLS + req_feats

    cols_match = False
    if table_exists(table_name, uat=uat):
        curr_cols = get_table_columns(table_name, uat=uat)
        # use sets as all cols should be unique names and this gens order
        cols_match = (set(curr_cols) == set(new_feat_cols))
    else:
        print("Table doesn't exist yet so fine to set new col schema")
        cols_match = True

    # if cols match then can go ahead and process
    if cols_match:
        df = handle_feats(feat_list, fixtures=fixtures)
    else:
        # for now given sqlite limitations, check if new feats > existing cols
        # if so then we delete the current features table and set new one
        if set(curr_cols).issubset(set(new_feat_cols)):
            print(
                'Old cols subset of new requested {} - deleting and recreating'.format(table_name))
            conn = create_conn(uat=uat)
            cur = conn.cursor()
            cur.execute('DROP TABLE {}'.format(table_name))
            conn.commit()
            conn.execute('VACUUM')
            conn.close()
            df = handle_feats(feat_list, fixtures=fixtures)
        else:
            print(
                'New requested cols do not match the existing {} columns'.format(table_name))
            return None

    # now we have a df for the new matches
    if df is None:
        # then no new matches and exit
        print('Exiting feature processing for table {}'.format(table_name))
        return None
    else:
        # we need to set the table down into sql
        try:
            conn = create_conn(uat=uat)
            df.to_sql(table_name, conn, if_exists='append', index=False)
        except:
            print('Failed to set down / append to {} table post calc'.format(table_name))

    return df


if __name__ == '__main__':
    None
