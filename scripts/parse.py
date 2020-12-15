'''
07-Dec-2020
Author: Mark Jamison
Script to scrape football match result and odds data
Maintains register for daily efficient scraping of data
Source Website: https://www.football-data.co.uk/
'''

from epl.features_parse import process_feature_data
from epl.matches_parse import process_fixture_data, process_match_data


# define feats
# we define the new base col name,the required cols and how to convert to t and opp equiv cols
FEATS = {'GF': {'Home': 'FTHG', 'Away': 'FTAG'},
         'GA': {'Home': 'FTAG', 'Away': 'FTHG'},

         'GFH': {'Home': 'FTHG'},
         'GAH': {'Home': 'FTAG'},
         'GFA': {'Away': 'FTAG'},
         'GAA': {'Away': 'FTHG'},

         'SF': {'Home': 'HS', 'Away': 'AS'},
         'SA': {'Home': 'AS', 'Away': 'HS'},

         'SFH': {'Home': 'HS'},
         'SAH': {'Home': 'AS'},
         'SFA': {'Away': 'AS'},
         'SAA': {'Away': 'HS'},

         'STF': {'Home': 'HST', 'Away': 'AST'},
         'STA': {'Home': 'AST', 'Away': 'HST'},

         'STFH': {'Home': 'HST'},
         'STAH': {'Home': 'AST'},
         'STFA': {'Away': 'AST'},
         'STAA': {'Away': 'HST'},

         'PPG': {'Home': 'FTR', 'Away': 'FTR'},
         'PPGH': {'Home': 'FTR'},
         'PPGA': {'Away': 'FTR'},
         }

FEAT_LIST = [{'feat_type': 'avg',
              'feat_dict': FEATS,
              'streak': x,
              'avg_type': 'Avg'} for x in [3, 5, 10, 20, 40, 80]]

if __name__ == '__main__':
    process_match_data()
    process_fixture_data()
    process_feature_data(FEAT_LIST, fixtures=False)
    process_feature_data(FEAT_LIST, fixtures=True)
