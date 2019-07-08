# Import the needed libraries
import pandas as pd
import numpy as np

# Load RAW data
raw_name_basics = pd.read_csv("imdb.name.basics.csv.gz")
raw_title_akas = pd.read_csv("imdb.title.akas.csv.gz")
raw_title_basics = pd.read_csv("imdb.title.basics.csv.gz")
raw_title_crew = pd.read_csv("imdb.title.crew.csv.gz")
raw_title_principals = pd.read_csv('imdb.title.principals.csv.gz')
raw_title_ratings = pd.read_csv('imdb.title.ratings.csv.gz')
raw_movie_info = pd.read_csv('rt.movie_info.tsv.gz', delimiter='\t')
raw_reviews = pd.read_csv('rt.reviews.tsv.gz', delimiter='\t', encoding='latin-1')
raw_movies = pd.read_csv('tmdb.movies.csv.gz')
raw_movie_budgets = pd.read_csv('tn.movie_budgets.csv.gz')

# Copies of raw data frames, to be cleaned below, then joined into data frame "movies"
budgeted_movies = raw_movie_budgets.copy()
genred_movies = raw_title_basics.copy()

# Clean budgeted_movies:
#           convert type, where appropriate: string (with $xxx,xxx) --> integer
#           rename columns where helpful
#           create new features: ROI and foreign gross
#           remove movies with zero revenue (assumed to be missing dates)
#           save copy of above to budgeted_movies_all_dates
#           restrict release date to 2010-2018 (inclusive)

# Convert strings in budget columns to integers; create columns with shorter names to be dropped below.
budgeted_movies['gross'] = pd.to_numeric(
    raw_movie_budgets.worldwide_gross.str.replace('$', '').str.replace(',', ''))
budgeted_movies['budget'] = pd.to_numeric(
    raw_movie_budgets.production_budget.str.replace('$', '').str.replace(',', ''))
budgeted_movies['domestic_gross'] = pd.to_numeric(
    raw_movie_budgets.domestic_gross.str.replace('$', '').str.replace(',', ''))

# Rename columns by copying then dropping all redundant columns, including those defined above.
budgeted_movies['title'] = budgeted_movies.movie
budgeted_movies.drop(['production_budget', 'worldwide_gross', 'movie'], axis=1)

# Create new budget features: ROI and Foreign Gross
budgeted_movies['roi'] = (budgeted_movies.gross - budgeted_movies.budget) / budgeted_movies.budget
budgeted_movies['foreign_gross'] = budgeted_movies.gross - budgeted_movies.domestic_gross

# Keep only movies with positive (i.e. nonzero) revenue (ROI > -1) (otherwise data assumed missing).
mask_positive_revenue = budgeted_movies['gross'] > 0
budgeted_movies = budgeted_movies[mask_positive_revenue]

# Save copy of data frame to budgeted_movies_all_dates so that original can be date restricted.
budgeted_movies_all_dates = budgeted_movies.copy()

# Keep only budgeted movies with release_dates between 2010 and 2018 (inclusive)
mask_after_2010 = pd.to_datetime(budgeted_movies.release_date) >= '2010-01-01'
mask_before_2019 = pd.to_datetime(budgeted_movies.release_date) < '2019-01-01'
budgeted_movies = budgeted_movies[mask_after_2010 & mask_before_2019]


# Clean genred_movies:
#           replace missing genre labels with "Unknown"
#           save copy of above to all_genred_movies
#           remove movies with Unknown genres
#           remove all movies from genres having sample size less than 10

# INSERT HERE FROM CODE BELOW

# Join budgeted and genres data frame: set index of each then join.
budgeted_movies.index = budgeted_movies['title']
genred_movies.index = genred_movies['primary_title']
movies = budgeted_movies.join(genred_movies, how='inner')

# NOW DERIVE genres data frame where each record is one genre.  Each record of movies is one movie.
# Include genre, sample_size, median_roi, proportion_failed, adjusted_risk, BtoR
# Could also include other summary statistics for each genre: mean, Q1, Q3, min, max
# Here proportion failed = P0, could also have P1, P10.
# Might also have an all_genres dataframe that is not sample size restricted.
# Sort genres by median ROI
test_median = movies.groupby('genres').roi.median().sort_values(ascending=False)

# Sample size is of concern. We want to drop any genre with less than 10 movies
# Counting the number of movies in each genre
test_count = movies.groupby('genres').roi.count().sort_values(ascending=False)
count_df = pd.DataFrame(test_count)
median_df = pd.DataFrame(test_median)
joined_genre_df = median_df.join(count_df, how='inner', lsuffix='median', rsuffix='count')


# Ok, sort by genre by count
joined_genre_df.sort_values(by='roicount', ascending=False) 

# Only keep genres that have more than 10 movies
good_genres = joined_genre_df[joined_genre_df['roicount'] >= 10]
good_genres = good_genres.sort_values(by='roimedian', ascending=False)

# Making a list of good genres to use
list_of_good_genres = list(good_genres.index)

# Select these genres out of our data frame
index_list = []
for index, element in enumerate(movies['genres']):
    if element in list_of_good_genres:
        index_list.append(index)


good_sample_df = movies.iloc[index_list, :]

# Dealing with duplicate movies
good_sample_df.index.duplicated().sum()
# There are 309 duplications


# Getting rid of duplicates
good_sample_df2 = good_sample_df.copy()  # Backup copy just in case
good_sample_df2.drop_duplicates(subset='movie', keep='first', inplace=True)

# Dropping duplicate columns
good_sample_df2.drop(columns=['movie', 'primary_title', 'original_title'], inplace=True)

genre_df = good_sample_df2.copy()
movies_df = good_sample_df2.copy()  # Now the dataframe is ready


# Adding dataframe coded for scatterplot roi to budget to profit comparison
# Joe's edits
# Similar date filter for df_movie_budgets for the roi/budget/profit scatterplot,
# built on same code, copy df created to avoid conflict if variable names changed
date_mask = (pd.to_datetime(raw_movie_budgets.release_date) >= '2010-01-01') & \
            (pd.to_datetime(raw_movie_budgets.release_date) < '2019-01-01')
df_movie_budgets_clip = raw_movie_budgets[date_mask]

# df_movie_budgets_clip['worldwide_gross_numeric']
#  = pd.to_numeric(df_movie_budgets_clip.worldwide_gross.str.replace('$','').str.replace(',',''))
# df_movie_budgets_clip['budget_numeric']
#  = pd.to_numeric(df_movie_budgets_clip.production_budget.str.replace('$','').str.replace(',',''))
df_movie_budgets_clip['profit'] = df_movie_budgets_clip['worldwide_gross'] - df_movie_budgets_clip['production_budget']

df_movie_budgets_clip.rename({'id': 'movie'}, inplace=True)

df_movie_budgets_clip['roi'] = df_movie_budgets_clip['profit']/df_movie_budgets_clip['production_budget']
df_movie_budgets_clip.join(raw_name_basics, how='inner')
df_movie_budgets_clip.set_index('movie', inplace=True)
df_movie_budgets_clip['roi'].sort_values(ascending=False)
genred_movies.set_index('primary_title', inplace=True)

# Create the_df which is where ROI/Budget/Profit pulls from
a_df = df_movie_budgets_clip.join(genred_movies, how='inner')
a_df = a_df.loc[~a_df.index.duplicated(keep='first')]

# Create subdataframes for ROI/Budget/Profit plot
#    (plot dataframe is gross_roi_max_df) and then combines them into gross_roi_max_df
top_5_budget = a_df.sort_values(by='production_budget', ascending=False, axis=0).iloc[:10, :]
top_5_budget['top_X_parameter'] = 'production_budget'
top_5_profit = a_df.sort_values(by='profit', ascending=False, axis=0).iloc[:10, :]
top_5_profit['top_X_parameter'] = 'gross'
top_5_roi = a_df.sort_values(by='roi', ascending=False, axis=0).iloc[:10, :]
top_5_roi['top_X_parameter'] = 'roi'

# Stack the dataframes
transition1_df = top_5_budget.append(top_5_profit)
gross_roi_max_df = transition1_df.append(top_5_roi)
gross_roi_max_df = gross_roi_max_df.reset_index()

# Creating indicator column for films that made money vs those that did not - will use for hue parameter in plot below
gross_roi_max_df['made_a_profit'] = 1
for row in range(0, gross_roi_max_df.shape[0]):
    if gross_roi_max_df.profit[row] > 0:
        gross_roi_max_df.made_a_profit[row] = 1
    else:
        gross_roi_max_df.made_a_profit[row] = 0
# rename column 'index' as title
gross_roi_max_df.rename(columns={'index': 'title'}, inplace=True)

# First, finding benefit
benefit = genre_df.groupby('genres').median()['roi'] 

# Now finding risk
total_genre_count = genre_df['genres'].value_counts()  # Good
total_genre_count_and_one = total_genre_count + 1  # Good
# Adding 1 to avoid divsion by zero (incase a certain genre did not lose money)
# Action,Adventure,Sci-Fi and Action,Adventure,Thriller have no failed profits. Scaling all movies up by 1.
failed_in_genre_count = genre_df[genre_df['roi'] < 0].groupby('genres').count()['roi'] + 1
failed_in_genre_count['Action,Adventure,Sci-Fi'] = 1
failed_in_genre_count['Action,Adventure,Thriller'] = 1

risk = np.array(failed_in_genre_count)/np.array(total_genre_count_and_one)

# now we have our Benefit-to-risk ratio
b2r = benefit/risk

# First, find benefit (median ROI)
median_roi = movies_df.groupby('genres').median()['roi'].to_frame()
median_roi.columns = ['median_roi']
# Now count the number of losers in each genre, if zero won't have a row for that genre
losers = movies_df[movies_df['roi'] < 0].groupby('genres').count()['roi'].to_frame()
losers.columns = ['losers']
total = movies_df.groupby('genres').count()['roi'].to_frame()
total.columns = ['total']
genres_df = median_roi.join(total, how='outer').join(losers, how='left').fillna(0)
# Now find the adjusted risk (losers + 1)/(total + 1) and BtoR
genres_df['adjusted_risk'] = (genres_df.losers + 1)/(genres_df.total + 1)
genres_df['BtoR'] = genres_df.median_roi/genres_df.adjusted_risk
genres_df.sort_values('BtoR', ascending=False)

# Get genres with small sample sizes
genres_to_drop = genres_df[genres_df['total'] < 10]


# Drop movies from genres with small samples
bigsample_movies = movies_df.copy()
bigsample_movies['title'] = bigsample_movies.index
bigsample_movies.set_index('genres', inplace=True, drop=False)
bigsample_movies.drop(genres_to_drop.index, axis=0, inplace=True)
bigsample_movies.set_index('title', inplace=True)
# Drop genres with small samples
bigsample_genres = genres_df.copy()
bigsample_genres.drop(genres_to_drop.index, inplace=True)
# rank the genres by BtoR
ranked_genres = list(bigsample_genres.sort_values('BtoR', ascending=False).index)

# Get a list of top 5 genres
top5_genres = ranked_genres[:5]

i_top = []
for i, element in enumerate(bigsample_movies['genres']):
    if element in top5_genres:
        i_top.append(i)

top5_df = bigsample_movies.iloc[i_top, :]


# Preparing to plot roi vs budget by genre
horror_df = movies_df[movies_df['genres'] == 'Horror,Mystery,Thriller']
# Action,Adventure,Sci-Fi = AASF
AASF_df = movies_df[movies_df['genres'] == 'Action,Adventure,Sci-Fi']
# Adventure,Animation,Comedy = AAC
AAC_df = movies_df[movies_df['genres'] == 'Adventure,Animation,Comedy']
# Comedy,Romance 
CR_df = movies_df[movies_df['genres'] == 'Comedy,Romance']
# Comedy,Drama,Romance
CDR_df = movies_df[movies_df['genres'] == 'Comedy,Drama,Romance']
