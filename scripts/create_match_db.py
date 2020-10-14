'''
06-Oct-2020
Author: Mark Jamison
Script to scrape football match result and odds data
Source Website: https://www.football-data.co.uk/
'''

# mixture of libs for web scraping, parsing and pandas
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


# define the site root
SITE_ROOT = 'https://www.football-data.co.uk/'
DATA_ROOT = 'https://www.football-data.co.uk/data.php'

# short function to standard dates in str format across 2 different formats


def standardise_dates(d):

    if len(d) == len('01/02/2000'):
        return pd.to_datetime(d, format='%d/%m/%Y')
    elif len(d) == len('01/02/20'):
        return pd.to_datetime(d, format='%d/%m/%y')
    else:
        return pd.to_datetime(d)


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

# get links that end '*m.php' and are not https links
countries = [x[:-5] for x in links if x[-5:] == 'm.php' and 'https' not in x]
# print('Countries where we have the data: {}'.format(', '.join(countries)))

# form the data links and then zip into a dictionary
country_links = [SITE_ROOT+x+'m.php' for x in countries]
country_dict = dict(zip(countries, country_links))
for c, l in country_dict.items():
    print("{}: {}".format(c, l))

# define individual csvs
all_links = {}

for country, link in country_dict.items():
    # get the page and parse into soup object
    req = Request(link)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")

    # get all the links on the data page
    csv_links = []
    for url_link in soup.findAll('a'):
        # get the label e.g. 'Premier League'
        label = url_link.contents[0]
        # get the link ref e.g. 'mmz4281/2021/E0.csv'
        l = url_link.get('href')
        # if link not null and is a csv then add it
        if l != None and '.csv' in l:
            csv_links.append([label, l])

    all_links[country] = csv_links

output_dfs = []

# for each season / league we have a link for
for country, links in all_links.items():
    for s in links:
        # form the query url
        query_url = SITE_ROOT + s[1]
        # format the season e.g. '19/20' into '2019/2020'
        season = s[1].split("/")[-2]
        if (2000+int(season[:2])) > dt.date.today().year:
            season = '19' + season[:2] + '/' + '19' + season[-2:]
        else:
            season = '20' + season[:2] + '/' + '20' + season[-2:]
        print("Querying url: {} for country: {}, league: {} and season: {}".format(
            query_url, country, s[0], season))
        # query it
        res = requests.get(query_url)
        # if good response, extract
        if res.status_code == 200:
            output = res.content
            df = pd.read_csv(io.StringIO(output.decode('utf-8', errors='ignore')),
                             parse_dates=True, error_bad_lines=False, warn_bad_lines=False)
            # add columns that define the season and league
            df['Country'] = country
            df['League'] = s[0]
            df['Season'] = season
            # add to the list of output dfs
            output_dfs.append(df)

# concat the resulting dataframes together
output = pd.concat(output_dfs)
print(output.head())

# get rid of cols with more than 99% nulls
df = output[output.columns[output.isnull().mean() < 0.99]]

# get rid of null divs as this means a data error
df = df[~df['Div'].isna()]

# fix types
df['Date'] = df.Date.apply(lambda x: standardise_dates(x))
df['PSCH'] = pd.to_numeric(df['PSCH'])

# final print pre write down
print(df.head())

try:
    # create dir name
    dir_name = 'data'
    # get the location of this script and strip the filename and dir
    script_location = os.path.realpath(__file__)
    root_dir = script_location.split('/')[:-2]
    db_dir = "/".join(root_dir + [dir_name])

    # now we have the data dir path, if not exist then make it
    if not os.path.exists(db_dir):
        print("database directory does not exist, creating now at: {}".format(db_dir))
        os.makedirs(db_dir)

    # create database file name
    db_name = 'match_results.sqlite'
    db_loc = "/".join([db_dir, db_name])

    # try to connect and save down data
    print("Trying to connect to: {}".format(db_loc))
    conn = sqlite3.connect(db_loc)
    df.to_sql('matches', conn, index=False)
    conn.close()
except:
    print("Unable to establish connection")


if __name__ == '__main__':
    None
