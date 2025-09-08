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
#turning our dates into a canonical stirng for datetime accessors etc.
#format is the format of our input string, errors=coerce means treat as a Nat, not a time (like Nan)

season_and_postseason['Month'] = season_and_postseason['Date'].dt.month
season_and_postseason['Day'] = season_and_postseason['Date'].dt.day

season_and_postseason['Alter Month'] = season_and_postseason['Month'].isin([1,2]) 



print(season_and_postseason.head())

