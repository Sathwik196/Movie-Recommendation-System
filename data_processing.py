import pandas as pd

# Load the dataset
netflix_data = pd.read_csv("film_data.csv")

# Handle missing values
netflix_data.fillna(value=0, inplace=True)

# Remove duplicate rows
netflix_data.drop_duplicates(inplace=True)
