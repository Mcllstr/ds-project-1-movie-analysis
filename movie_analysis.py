# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Load all available data 
#bom_movie_gross_df = pd.read_csv('bom.movie_gross.csv.gz')
#imdb_name_basics_df = pd.read_csv('imdb.name.basics.csv.gz')
#imdb_title_akas_df = pd.read_csv('imdb.title.akas.csv.gz')
#imdb_title_crew_df = pd.read_csv('imdb.title.crew.csv.gz')
#imdb_title_principals_df = pd.read_csv('imdb.title.principals.csv.gz')
#imdb_title_ratings_df = pd.read_csv('imdb.title.ratings.csv.gz')
#rt_movie_info_df = pd.read_csv('rt.movie_info.tsv.gz', sep='\t')
#rt_reviews_df = pd.read_csv('rt.reviews.tsv.gz', sep ='\t', encoding = 'latin-1')
#tmdb_movies_df = pd.read_csv('tmdb.movies.csv.gz')


# These are the two csv's we use
tn_movie_budgets_df = pd.read_csv('tn.movie_budgets.csv.gz')
imdb_title_basics_df = pd.read_csv('imdb.title.basics.csv.gz')

# Check out tn_movie_budgets_df
tn_movie_budgets_df.head()

# Convert gross and production columns in tn_movie_budgets_df from str to int
gross = pd.to_numeric(tn_movie_budgets_df.worldwide_gross.str.replace('$','')).replace(',','')
budget = pd.to_numeric(tn_movie_budgets_df.production_budget.str.replace('$','').str.replace(',',''))

tn_movie_budgets_df['worldwide_gross'] = gross
tn_movie_budgets_df['production_budget'] = budget

# Calculate and create an return on investment (ROI) column
roi = (gross - budget)/budget

tn_movie_budgets_df['roi'] = roi



# Grab the movies that are in between 2010 and 2018

# NOTE FOR TEAMMATES: The backslash here allows you to break up the line\
# when it gets too long, which is something I just learned
mask = (pd.to_datetime(tn_movie_budgets.release_date) >= '2010-01-01') & (pd.to_datetime(tn_movie_budgets.release_date) < '2019-01-01')

tn_movie_budgets_to_date = tn_movie_budgets_df[mask]





