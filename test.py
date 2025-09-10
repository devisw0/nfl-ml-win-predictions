import pandas as pd
import numpy as np

def tests():
    df = pd.read_csv('2017-2025_scores.csv')

    print((df['HomeScore']).dtype)
    print((df['AwayScore'].dtype))



if __name__ == "__main__":
    tests()
