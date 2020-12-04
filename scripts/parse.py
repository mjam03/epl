'''
02-Dec-2020
Author: Mark Jamison
Script to scrape football match result and odds data
Maintains register for daily efficient scraping of data
Source Website: https://www.football-data.co.uk/
'''

from epl.matches_parse import process_fixture_data, process_match_data

if __name__ == '__main__':
    process_match_data()
    process_fixture_data()
