'''
02-Dec-2020
Author: Mark Jamison
Script to scrape football match result and odds data
Maintains register for daily efficient scraping of data
Source Website: https://www.football-data.co.uk/
'''

from bs4 import BeautifulSoup
import datetime as dt
import io
import json
import numpy as np
import os
import pandas as pd
import requests
import sqlite3
from urllib.request import Request, urlopen


from epl.query import create_conn, query_creator, query_db, table_exists

# define the site root
SITE_ROOT = 'https://www.football-data.co.uk/'
DATA_ROOT = 'https://www.football-data.co.uk/data.php'
FIXTURES_ROOT = 'https://www.football-data.co.uk/matches.php'
DB_NAME = 'footie.sqlite'
DB_NAME_UAT = 'footie_uat.sqlite'


def get_root_dir():

    script_location = os.path.realpath(__file__)
    root_dir = script_location.split('/')[:-2]
    return '/'.join(root_dir)


def get_reg_filename(reg_name):
    # get reg location
    reg_dir = '/'.join([get_root_dir(), 'data'])
    reg_file = '/'.join([reg_dir, reg_name])+'.csv'
    return reg_file


def standardise_dates(d):

    if len(d) == len('01/02/2000'):
        return pd.to_datetime(d, format='%d/%m/%Y')
    elif len(d) == len('01/02/20'):
        return pd.to_datetime(d, format='%d/%m/%y')
    else:
        return pd.to_datetime(d)


def get_country_urls():
    '''
            Returns dict of {country: data_url}
            '''
    # get the page and parse into soup object
    req = Request(DATA_ROOT)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")

    # get all the links on the data page
    links = []
    for link in soup.findAll('a'):
        l = link.get('href')
        if l != None:
            links.append(l)

    # now we need to get the list of links that link to pages for data
    # this involves parsing the page for the country name
    # from inspection these pages end '*m.php'
    countries = [x[:-5]
                 for x in links if x[-5:] == 'm.php' and 'https' not in x]

    # form the data links and then zip into a dictionary
    country_links = [SITE_ROOT+x+'m.php' for x in countries]
    country_dict = dict(zip(countries, country_links))
    return country_dict


def get_most_recents(country_dict):
    '''
    country_dict: dict of {country, data_url} to iterate through
    Returns dataframe of Date / Country / MostRecentSeason
    '''
    # dict of country key to most recent season id value
    most_recents = {}
    # get todays date
    d = dt.date.today()

    for country, link in country_dict.items():
        # get the page and parse into soup object
        req = Request(link)
        html_page = urlopen(req)
        soup = BeautifulSoup(html_page, "lxml")

        # given page is rev chronological i.e. most recent season first
        # can grab first valid link and strip season id from that
        for url_link in soup.findAll('a'):
            # if we haven't already found the most recent season
            if country not in most_recents.keys():
                # get the link ref e.g. 'mmz4281/2021/E0.csv'
                l = url_link.get('href')
                # if link not null and is a csv then is valid data link
                if l != None and '.csv' in l:
                    season_id = l.split('/')[1]
                    # create record for table creation
                    rec = {'Date': d, 'Country': country,
                           'MostRecentSeason': season_id}
                    most_recents[country] = rec

    # convert this into a dataframe and return
    df_recents = pd.DataFrame(most_recents.values())
    return df_recents


def update_most_recents_register(uat=False):
    '''
    Fetches latest most recents, creates/appends to most_recents_register
    '''
    # used for updating reg if already exists (re-running on same day)
    d = pd.to_datetime(dt.date.today())
    # get country urls
    country_dict = get_country_urls()
    # iterate through to get mostrecent season id per country
    df_recents = get_most_recents(country_dict)

    # dir structure:
    # /scripts
    #		/this_script.py
    # /data
    #		/most_recents_register.csv
    dir_name = 'data'
    # get the location of this script and strip the filename and dir
    root_dir = get_root_dir()
    reg_dir = "/".join([root_dir, dir_name])

    # first check if dir 'data' exists - if not create it
    if not os.path.exists(reg_dir):
        print("database directory does not exist, creating now at: {}".format(reg_dir))
        os.makedirs(reg_dir)

    # now check if register file exists
    if uat:
        reg_file = "/".join([reg_dir, 'register_most_recents_uat.csv'])
    else:
        reg_file = "/".join([reg_dir, 'register_most_recents.csv'])

    if not os.path.exists(reg_file):
        # doesn't exist - just set down what we have without index
        try:
            print("register_most_recents.csv doesn't exist yet - writing it")
            df_recents.to_csv(reg_file, index=False)
        except:
            print("Unable to write register_most_recents.csv for first time")
    else:
        try:
            # register exists so we want to load it, append and set
            df_curr_recents = pd.read_csv(reg_file, parse_dates=['Date'])
            # remove today's date if there - must be re-running and want to have unique per date
            df_curr_recents = df_curr_recents[df_curr_recents.Date != d]
            # concat and reindex
            df_reg = pd.concat([df_curr_recents, df_recents])
            df_reg = df_reg.reset_index(drop=True)
            print("Writing down regsiter_most_recents{}.csv post append".format(
                ('_uat' if uat else '')))
            df_reg.to_csv(reg_file, index=False)
        except:
            print("Unable to append to register_most_recents{}.csv".format(
                ('_uat' if uat else '')))

    return reg_file


def get_register(reg_name):
    '''
    Returns pd.DataFrame of the register requested
    '''
    # get reg location
    reg_file = get_reg_filename(reg_name)
    # if doesn't exist then report
    if not os.path.exists(reg_file):
        print('Register: {} does not exist yet at loc: {}'.format(reg_name, reg_file))
        # return None so can use output to fire if statements
        return None

    df_reg = pd.read_csv(reg_file, parse_dates=True)
    for c in ['Date']:
        if c in df_reg.columns:
            df_reg[c] = pd.to_datetime(df_reg[c])
    # convert season col to string as pd auto imports as int
    for c in ['MostRecentSeason', 'Season', 'ParseMessage']:
        if c in df_reg.columns:
            df_reg[c] = df_reg[c].apply(str)
    if 'ParseMessage' in df_reg.columns:
        df_reg['ParseMessage'] = np.where(
            df_reg['ParseMessage'] == 'nan', '', df_reg['ParseMessage'])
    return df_reg


def get_all_curr_urls(country_dict):
    '''
    Returns pd.DataFrame of all the current data links for country_dict fed
    '''
    # get today date
    d = pd.to_datetime(dt.date.today())
    # define array to append link dicts to
    csv_links = []
    # iterate through countries
    for country, link in country_dict.items():

        # handle html data into python data structure
        req = Request(link)
        html_page = urlopen(req)
        soup = BeautifulSoup(html_page, "lxml")

        # get all the links on the data page
        for url_link in soup.findAll('a'):
            # get the label e.g. 'Premier League'
            label = url_link.contents[0]
            # get the link ref e.g. 'mmz4281/2021/E0.csv'
            l = url_link.get('href')
            # if link not null and is a csv then add it
            if l != None and '.csv' in l:
                # then construct the rec and append
                rec = {'Date': d,
                       'Country': country,
                       'DivName': label,
                       'Div': l.split('/')[-1][:-4],
                       'Season': l.split('/')[-2],
                       'url': SITE_ROOT+l}

                csv_links.append(rec)
    df_links = pd.DataFrame(csv_links)
    return df_links


def get_fixtures_link():
    # get today date
    d = pd.to_datetime(dt.date.today())

    req = Request(FIXTURES_ROOT)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")

    # find the fixtures csv link
    for url_link in soup.findAll('a'):
        # get the link ref e.g. 'mmz4281/2021/E0.csv'
        l = url_link.get('href')
        # if link not null and is a csv then add it
        if l == 'fixtures.csv':
            rec = {'Date': d,
                   'url': SITE_ROOT+l}

    return pd.DataFrame([rec])


def create_match_register(uat=False):
    '''
    Returns pd.DataFrame for latest register for matches from football-data.co.uk
    '''
    # get country dicts and recent urls
    country_dict = get_country_urls()
    df_links = get_all_curr_urls(country_dict)
    # get recents register
    if uat:
        df_recents = get_register('register_most_recents_uat')
    else:
        df_recents = get_register('register_most_recents')
    # create col to denote most recent season
    df_links = pd.merge(left=df_links, right=df_recents,
                        how='left', on=['Date', 'Country'])
    df_links['IsMostRecent'] = (
        df_links['Season'] == df_links['MostRecentSeason'])
    df_links = df_links.drop(columns=['MostRecentSeason'])

    # update status as 'New' and empty parse message (for now)
    df_links['Status'] = 'New'
    df_links['ParseMessage'] = ''

    # now grab the existing register if it exists
    if uat:
        df_reg = get_register('register_matches_uat')
        reg_file = get_reg_filename('register_matches_uat')
    else:
        df_reg = get_register('register_matches')
        reg_file = get_reg_filename('register_matches')
    print(reg_file)
    if df_reg is None:
        # if none then does not exist - so we create and set what we have
        print('No reg detected - creating new reg in: {}'.format(reg_file))
        df_links.to_csv(reg_file, index=False)
        df_new_reg = df_links.copy()
    else:
        # then reg exists and we need to update it
        # keep links if they are most recent on this date only
        d = df_links.Date.max()
        df_old_recs = df_reg[(df_reg.IsMostRecent) & (
            df_reg.Date == d) & (df_reg.Status == 'Processed')]
        # now we only keep most recents if they aren't in df_old_recs
        df_recs = df_links[df_links['IsMostRecent']]
        df_recs = df_recs[~df_recs.url.isin(df_old_recs.url)]

        # we also keep those which have never appeared before based on url
        df_new = df_links[~df_links.url.isin(df_reg.url)]
        # now we concat them and remove any potential dupes
        df_new_reg = pd.concat([df_recs, df_new]).drop_duplicates()
        df_new_reg = pd.concat([df_reg, df_new_reg])
        print('Setting new reg down into: {}'.format(reg_file))
        df_new_reg.to_csv(reg_file, index=False)

    return df_new_reg


def update_fixture_register(uat=False):

    # get fixture csv link in df format
    df_links = get_fixtures_link()

    # add parse status
    df_links['Status'] = 'New'
    df_links['ParseMessage'] = ''

    # get current reg (if it exists)
    if uat:
        df_reg = get_register('register_fixtures_uat')
        reg_file = get_reg_filename('register_fixtures_uat')
    else:
        df_reg = get_register('register_fixtures')
        reg_file = get_reg_filename('register_fixtures')
    print(reg_file)

    if df_reg is None:
        # if none then does not exist - so we create and set what we have
        print('No reg detected - creating new reg in: {}'.format(reg_file))
        df_links.to_csv(reg_file, index=False)
        df_new_reg = df_links.copy()
    else:
        # then reg exists and we need to update it
        # simple here - just concat on the end
        d = pd.to_datetime(dt.date.today())
        if len(df_reg[(df_reg.Date == d) & (df_reg.Status == 'Processed')]) > 0:
            # then we have already processed today, do nothing
            df_new_reg = df_reg.copy()
        else:
            df_new_reg = pd.concat([df_reg, df_links])
            df_new_reg = df_new_reg.drop_duplicates()
            print('Setting new reg down into: {}'.format(reg_file))
            df_new_reg.to_csv(reg_file, index=False)

    return df_new_reg


def get_new_files(table_name, uat=False):
    '''
    Returns pd.DataFrame of files to be parsed for a given register
    '''
    reg_name = 'register_' + table_name + ('_uat' if uat else '')
    df_reg = get_register(reg_name)
    if df_reg is None:
        # then no files so end here
        print('No register exists yet for table: {}'.format(table_name))
        return None
    else:
        # register exists - get non-processed files - both 'New' and 'Error'
        df_new = df_reg[df_reg.Status != 'Processed']

    return df_new


def fetch_file(url):

    # query it
    print('Fetching {}'.format(url))
    res = requests.get(url)
    # if good response, extract
    if res.status_code == 200:
        output = res.content
        df = pd.read_csv(io.StringIO(output.decode('utf-8', errors='ignore')),
                         parse_dates=True, error_bad_lines=False, warn_bad_lines=False)
        return df
    else:
        print('Bad response code from {} for {}'.format(SITE_ROOT, url))
        return None


def fetch_new_files(df_new):
    '''
    Accept pd.DataFrame of new files, fetches them and cleans them
    '''
    # add a new col which will be the resulting dataframe
    df_res = df_new.copy()
    df_res['Results'] = None

    for index, row in df_new.iterrows():
        try:
            df = fetch_file(row.url)
            # add new cols and keep col order if they are in reg (not in fixture reg)
            if any([x in row.index for x in ['Country', 'DivName', 'Season']]):
                cols = list(df.columns)
                df['Country'] = row.Country
                df['League'] = row.DivName
                df['Season'] = row.Season
                df = df[['Country', 'League', 'Season'] + cols]
            # remove cols with more than 99% nulls
            df = df[df.columns[df.isnull().mean() < 0.99]]

            df_res.at[index, 'Results'] = df
            df_res.at[index, 'Status'] = 'Processed'
        except:
            # if can't then log and update the reg
            print('Unable to fetch file: {}'.format(row.url))

            df_res.at[index, 'Results'] = None
            df_res.at[index, 'Status'] = 'Error'
            df_res.at[index, 'ParseMessage'] = 'ErrorOnRequest'

    return df_res


def update_register(reg_name, new_reg):
    '''
    Queries current register and updates with new data
    '''
    # get current reg
    reg_file = get_reg_filename(reg_name)
    reg = get_register(reg_name)
    reg.update(new_reg)
    reg.to_csv(reg_file, index=False)
    return reg


def handle_initial_match_db(df_new, uat=False):
    '''
    Function to handle initial creation of matches table in sqlite
    Handled separately in order to determine which columns to keep
    All updates then will be joins where columns must be strict subset of existing db
    i.e. can't have new columns created  - seems okay restriction as data feed relatively mature
    '''
    # concat all the res dfs together
    dfs = df_new[df_new.Status == 'Processed']
    df = pd.concat(list(dfs['Results']))
    # remove col if >99% of col are NaNs in combined df
    df = df[df.columns[df.isnull().mean() < 0.99]]
    df = clean_data(df)

    # set down data into db
    try:
        conn = create_conn(uat=uat)
        print('Connection established - setting down intial db')
        df.to_sql('matches', conn, index=False)
        conn.close()
        df_new['Status'] = 'Processed'
        df_new['ParseMessage'] = ''
    except:
        print('Unable to set down intial db')
        df_new['Status'] = 'Error'
        df_new['ParseMessage'] = 'Failed at initial setdown into sqlite'

    # update the register
    if uat:
        reg_name = 'register_matches_uat'
    else:
        reg_name = 'register_matches'
    new_reg = update_register(reg_name, df_new.drop(columns=['Results']))

    return df_new


def delete_table_rows(table_name, wc=None, uat=False):

    # create delete query
    sel_query = query_creator(table_name, wc=wc)
    del_query = sel_query.replace('SELECT *', 'DELETE')

    # establish conn and execute query
    try:
        conn = create_conn(uat=uat)
        cur = conn.cursor()
        cur.execute(del_query)
        conn.commit()
        conn.close()
        return True
    except:
        print('Failed to execute delete query: {}'.format(del_query))
        return False


def clean_data(df):

    # remove any cols with 'Unnamed' in them
    df = df[[x for x in df.columns if 'Unnamed' not in x]]
    # remove any rows if Div is na as garbage (as will all other cols)
    df = df[~df.Div.isna()]
    # standardise dates
    df['Date'] = df.Date.apply(lambda x: standardise_dates(x))
    return df


def handle_update_match_db(df_new, uat=False):

    # for each new entry
    for index, row in df_new.iterrows():
        # create new data
        new_data = row['Results']
        # create where clause that uniquely defines a data file
        wc = {'Country': ['=', row['Country']],
              'Div': ['=', row['Div']],
              'Season': ['=', row['Season']]}
        sel_query = query_creator('matches', wc=wc)
        # query existing data - to get cols
        old_data = query_db(sel_query, uat=uat)

        # only keep existing cols
        new_data = new_data[[
            x for x in new_data.columns if x in old_data.columns]]
        new_data = clean_data(new_data)

        # now we delete the old data and insert the new data
        old_del = delete_table_rows('matches', wc=wc, uat=uat)
        if not old_del:
            df_new.at[index, 'Status'] = 'Error'
            df_new.at[index, 'ParseMessage'] = 'Failed to delete old data'
        else:
            try:
                conn = create_conn(uat=uat)
                new_data.to_sql('matches', conn,
                                if_exists='append', index=False)
                conn.close()
                df_new.at[index, 'Status'] = 'Processed'
                df_new.at[index, 'ParseMessage'] = ''

            except:
                df_new.at[index, 'Status'] = 'Error'
                df_new.at[index, 'ParseMessage'] = 'Deleted old data, failed to insert new data'

        # update the register
    if uat:
        reg_name = 'register_matches_uat'
    else:
        reg_name = 'register_matches'
    new_reg = update_register(reg_name, df_new.drop(columns=['Results']))

    return df_new


def process_match_data(uat=False, div_list=None, season_list=None):
    '''
    Function that:
     - Gets most recent season and creates register
     - Gets urls and creates/updates file reg
     - Fetches new files and processes
     - If no db currently exists then joins all together and sets
     - If db exists then incrementally updates and removes new cols
     - Once done then update register to be processed
    '''
    # update most recents register
    update_most_recents_register(uat=uat)
    # update matches register and return it
    create_match_register(uat=uat)

    # restriction possibility for testing
    if (div_list is not None) or (season_list is not None):
        reg_name = 'register_matches'+('_uat' if uat else '')
        reg_file = get_reg_filename(reg_name)
        reg = get_register(reg_name)

        if (div_list is not None) & (season_list is not None):
            reg = reg[(reg.Div.isin(div_list)) & (
                reg.Season.isin(season_list))]
        elif div_list is not None:
            reg = reg[reg.Div.isin(div_list)]
        elif season_list is not None:
            reg = reg[reg.Season.isin(season_list)]
        reg.to_csv(reg_file, index=False)

    # get new files from reg
    df_new = get_new_files('matches', uat=uat)
    # fetch new files if poss
    df_new = fetch_new_files(df_new)
    if len(df_new) == 0:
        # no files to process
        print('No new files to process for matches table')
        return None
    # handle new files depending on whether or not db exists
    if uat:
        db_file = '/'.join([get_root_dir(), 'data', DB_NAME_UAT])
    else:
        db_file = '/'.join([get_root_dir(), 'data', DB_NAME])
    if (not table_exists('matches')) or (not os.path.exists(db_file)):
        # then db and table does not exist and this is inital set
        print('Database doesnt exist - going to create it now')
        res = handle_initial_match_db(df_new, uat=uat)
    else:
        # db exists and we need to update the data in the table
        res = handle_update_match_db(df_new, uat=uat)
    return res


def clean_and_join_fixture_data(df_new, uat=False):

    # add country, league and season data
    divs = query_db('SELECT Div, Country, League from matches GROUP BY Div',
                    uat=uat).sort_values('Country')
    # assume fixtures are most recent season
    seasons = get_register('register_most_recents' +
                           ('_uat' if uat else '')).drop(columns='Date')
    seasons = seasons.rename(columns={'MostRecentSeason': 'Season'})

    # clean the data
    res = df_new['Results'].values[0]
    res = clean_data(res)

    cols = list(res.columns)
    res = pd.merge(left=res, right=divs, how='left', on=['Div'])
    res = pd.merge(left=res, right=seasons, how='left', on=['Country'])
    res = res[['Country', 'League', 'Season'] + cols]

    df_new.at[df_new.index[0], 'Results'] = res

    return df_new


def handle_initial_fixture_db(df_new, uat=False):

    # add asof date col
    df = df_new['Results'].values[0]
    cols = list(df.columns)
    df['AsOfDate'] = pd.to_datetime(dt.date.today())
    df = df[['AsOfDate'] + cols]

    # set down data into db
    try:
        conn = create_conn(uat=uat)
        print('Connection established - setting down intial db')
        df.to_sql('fixtures', conn, index=False)
        conn.close()
        df_new['Status'] = 'Processed'
        df_new['ParseMessage'] = ''
    except:
        print('Unable to set down intial fixture db')
        df_new['Status'] = 'Error'
        df_new['ParseMessage'] = 'Failed at initial setdown into sqlite'

    # update the register
    if uat:
        reg_name = 'register_fixtures_uat'
    else:
        reg_name = 'register_fixtures'
    new_reg = update_register(reg_name, df_new.drop(columns=['Results']))

    return df_new


def handle_update_fixture_db(df_new, uat=False):

    # no updating required - merely appending on the end
    df = df_new['Results'].values[0]
    cols = list(df.columns)
    df['AsOfDate'] = pd.to_datetime(dt.date.today())
    df = df[['AsOfDate'] + cols]

    try:
        conn = create_conn(uat=uat)
        df.to_sql('fixtures', conn, if_exists='append', index=False)
        conn.close()
        df_new['Status'] = 'Processed'
        df_new['ParseMessage'] = ''
    except:
        print('Failed to append fixture data to table')
        df_new['Status'] = 'Error'
        df_new['ParseMessage'] = 'Failed to append fixture data'

    if uat:
        reg_name = 'register_fixtures_uat'
    else:
        reg_name = 'register_fixtures'
    new_reg = update_register(reg_name, df_new.drop(columns=['Results']))

    return df_new


def process_fixture_data(uat=False):

    # update the register
    update_fixture_register(uat=uat)

    # get new files
    d = pd.to_datetime(dt.date.today())
    df_new = get_new_files('fixtures', uat=uat)
    # for fixtures only get on same day
    df_new = df_new[df_new.Date == d]

    # parse
    df_new = fetch_new_files(df_new)
    if len(df_new) == 0:
        # no files to process
        print('No new files to process for fixtures table')
        return None

    # clean and join meta data
    df_new = clean_and_join_fixture_data(df_new)

    # update / create fixtures table
    if uat:
        db_file = '/'.join([get_root_dir(), 'data', DB_NAME_UAT])
    else:
        db_file = '/'.join([get_root_dir(), 'data', DB_NAME])
    if (not table_exists('fixtures')) or (not os.path.exists(db_file)):
        # then db and table does not exist and this is inital set
        print('fixtures table doesnt exist - going to create it now')
        res = handle_initial_fixture_db(df_new, uat=uat)
    else:
        print('fixtures table exists - appending to it now')
        res = handle_update_fixture_db(df_new, uat=uat)
    return res


if __name__ == '__main__':
    None
