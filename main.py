import pandas as pd
import numpy as np

df = pd.read_csv('2017-2025_scores.csv')

df = df.drop(['GameStatus','Day','AwayRecord','HomeRecord','AwaySeeding',	'HomeSeeding',
              'PostSeason'], axis = 1)

no_preseason = df[~df['Week'].str.contains('Preseason', case=False, na=False)]
#case is case sensitivity (true = case sensitive), ~ = not, na = False means we treat Nans as False so in our case they arent removed

season_and_postseason = no_preseason[~no_preseason['Week'].str.contains('Hall', case=False, na=False)]
#format is the format of our input string, errors=coerce means treat as a Nat, not a time (like Nan)
season_and_postseason = season_and_postseason.copy()  

season_and_postseason['Date'] = pd.to_datetime(
    season_and_postseason['Date'].str.strip(),
    format='%m/%d',
    errors='coerce'
)
#turning our dates into a canonical stirng for datetime accessors etc. such that it can put our dates in a way datetime can understand
#so after this line, we specify the format it is then able to understand all since it changes the way the dates are such that it can understand
#format is the format of our input string, errors=coerce means treat as a Nat, not a time (like Nan)

season_and_postseason['Month'] = season_and_postseason['Date'].dt.month
season_and_postseason['Day'] = season_and_postseason['Date'].dt.day

mask = season_and_postseason['Month'].isin([1,2]) 

season_and_postseason['game year'] = season_and_postseason['Season']

season_and_postseason.loc[mask, 'game_year'] = (
    season_and_postseason.loc[mask, 'Season'] + 1
)

game_date = {'year': season_and_postseason['game_year'], 'month': season_and_postseason['Month'], 'day': season_and_postseason['Day'] }

season_and_postseason['game_date'] = pd.to_datetime(game_date, errors = 'coerce')

season_and_postseason = season_and_postseason.dropna(subset = ['game_date'])

season_and_postseason = season_and_postseason.sort_values(by = 'game_date')

print(season_and_postseason.head())

