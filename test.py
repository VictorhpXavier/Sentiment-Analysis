import pandas as pd

# Load your datasets
reddit_df = pd.read_csv('../Dataset/Reddit_Data.csv')
twitter_df = pd.read_csv('../Dataset/Twitter_Data.csv')

# Print the columns to inspect
print("Reddit DataFrame columns:", reddit_df.columns)
print("Twitter DataFrame columns:", twitter_df.columns)
