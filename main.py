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

#making date column into a datetime object (str is an accessor)
#.astype converts series to specified type
season_and_postseason['Date'] = pd.to_datetime(
    season_and_postseason['Date'].astype(str).str.strip(),
    format='%m/%d',
    errors='coerce'
)
#turning our dates into a canonical stirng for datetime accessors etc. such that it can put our dates in a way datetime can understand
#so after this line, we specify the format it is then able to understand all since it changes the way the dates are such that it can understand
#format is the format of our input string, errors=coerce means treat as a Nat, not a time (like Nan)

#dt is an accessor
#just extracting values from our date tiem object
season_and_postseason['Month'] = season_and_postseason['Date'].dt.month
season_and_postseason['Day'] = season_and_postseason['Date'].dt.day

#making a boolean mask (same lenght as df and Trues or Falses only)
mask = season_and_postseason['Month'].isin([1,2]) 

#making copy to make edits
season_and_postseason['game_year'] = season_and_postseason['Season']

#we take the rows where condition is true take the values in the game_year column for those rows
#we then set them equal to the rows where the mask is true for the season column, add 1 and replace the ones on LHS (game_year column)
season_and_postseason.loc[mask, 'game_year'] = (
    season_and_postseason.loc[mask, 'Season'] + 1
)

#dictionary for date values
game_date = {'year': season_and_postseason['game_year'], 'month': season_and_postseason['Month'], 'day': season_and_postseason['Day'] }

#want to convert the dates to a date time object
#our keys year, month, day are a datetime case sepcific thing so they have to be the same always
season_and_postseason['game_date'] = pd.to_datetime(game_date, errors = 'coerce')

#dropping any nans in game date
season_and_postseason = season_and_postseason.dropna(subset = ['game_date'])

#sorting our df by the values in game_date column
season_and_postseason = season_and_postseason.sort_values(by = 'game_date')

#resetting our index after sorting the values
#drop = true since resetng index makes old index a column so we want to drop it
season_and_postseason = season_and_postseason.reset_index(drop = True)

#Just a mask, True if home win, false if else
season_and_postseason['home_win_binary'] = season_and_postseason['HomeScore'] > season_and_postseason['AwayScore']

#converting true and falses from prev line to 1s and 0s
season_and_postseason['home_win_binary'] = season_and_postseason['home_win_binary'].astype(int)

#finding score differences
season_and_postseason['score_difference'] = season_and_postseason['HomeScore'] - season_and_postseason['AwayScore']

#this column didnt have the year and is repetative
season_and_postseason = season_and_postseason.drop(columns=['Date'])

print(season_and_postseason.head())

print("Rows:", len(season_and_postseason))
print("Week NaNs:", season_and_postseason['Week'].isna().sum())
print(season_and_postseason['home_win_binary'].value_counts(normalize=True).rename('class_balance'))
#value counts counts the frequencies in a series 
#normalize = True turns the frequencies into percentages 

games = season_and_postseason.reset_index(names='game_id')
#resets index and moves the old one to a column named game_id

#we want to create home and away views such that we can create X data for the model
#we take the columns we want specifically and copy the data
#single brackets in pandas returns a series and duouble brakcers returns the dataframe
#this is because we are using the list (inner []) as the input for the df
home_view = games[['game_id','game_date','Season','HomeTeam','AwayTeam','HomeScore','AwayScore']].copy()
away_view = games[['game_id','game_date','Season','HomeTeam','AwayTeam','HomeScore','AwayScore']].copy()




home_view = home_view.rename(columns={'HomeTeam':'team','AwayTeam':'opponent',
                                      'HomeScore':'points_for','AwayScore':'points_against'},
                                      inplace=False)

away_view = away_view.rename(columns={'HomeTeam':'opponent','AwayTeam':'team',
                                      'HomeScore':'points_against','AwayScore':'points_for'},
                                      inplace=False)



home_view['is_home'] = 1
away_view['is_home'] = 0

game = pd.concat([home_view,away_view], axis = 0)