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


'''!!!!!!!!'''
# Joe, I think this block is what Sean is turning into a function
# Will take a series with strings + dollar signs and return a 
# series of integers.

# Convert gross and production columns in tn_movie_budgets_df from str to int
gross_without_dSign = tn_movie_budgets_df.worldwide_gross.str.replace('$','')
gross_without_comma = gross_without_dSign.str.replace(',', '')
gross = pd.to_numeric(gross_without_comma)

budget_without_dsign_and_comma = tn_movie_budgets_df.production_budget.str.replace('$','').str.replace(',','')
budget = pd.to_numeric(budget_without_dsign_and_comma)
'''!!!!!!!!!'''
# END Of FUNCTION 1. I AM GUESSING.


# Reassigning the columns in the data frame
tn_movie_budgets_df['worldwide_gross'] = gross
tn_movie_budgets_df['production_budget'] = budget



# Calculate and create an return on investment (ROI) column
roi = (gross - budget)/budget
tn_movie_budgets_df['roi'] = roi



# Grab the movies that are in between 2010 and 2018

# NOTE FOR TEAMMATES: The backslash here allows you to break up the line\
# when it gets too long, which is something I just learned
mask = (pd.to_datetime(tn_movie_budgets_df.release_date) >= '2010-01-01') & \
		(pd.to_datetime(tn_movie_budgets_df.release_date) < '2019-01-01')

tn_movie_budgets_to_date = tn_movie_budgets_df[mask]


# Join data frames on movie titles

# I don't know how to do this without setting the index to the
# columns you want to merge on. Adding the paramter on = ['col1', 'col2']
# gives me errors, but this way works.
tn_movie_budgets_to_date.index = tn_movie_budgets_to_date['movie']
imdb_title_basics_df.index = imdb_title_basics_df['primary_title']

the_df = tn_movie_budgets_to_date.join(imdb_title_basics_df, how ='inner')

# filling in missing roi data
for index, element in enumerate(the_df['roi']):
    if (element < -0.9999999) & (element > -1.01):
        the_df['roi'][index] = np.nan




        
median_genres = the_df.groupby('genres').roi.median().sort_values(ascending = False)
count_genres = the_df.groupby('genres').roi.count().sort_values(ascending = False)
count_df = pd.DataFrame(count_genres)
median_df = pd.DataFrame(median_genres)
joined_genre_df = median_df.join(count_df, how='inner', lsuffix='median', rsuffix='count')
joined_genre_df.sort_values(by='roicount', ascending=False)
genre_keepers = joined_genre_df[joined_genre_df['roicount'] >= 10]
print(genre_keepers.head())
