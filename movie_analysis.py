# Import the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
import seaborn as sns


sns.set()



# Importing Data into DFs
bom_movie_gross = pd.read_csv("bom.movie_gross.csv.gz")
imdb_name_basics = pd.read_csv("imdb.name.basics.csv.gz")
imdb_title_akas = pd.read_csv("imdb.title.akas.csv.gz")
imdb_title_basics = pd.read_csv("imdb.title.basics.csv.gz")
imdb_title_crew = pd.read_csv("imdb.title.crew.csv.gz")
imdb_title_principals = pd.read_csv('imdb.title.principals.csv.gz')
imdb_title_principals.head()
imdb_title_ratings = pd.read_csv('imdb.title.ratings.csv.gz')
rt_movie_info = pd.read_csv('rt.movie_info.tsv.gz',delimiter='\t')
rt_reviews = pd.read_csv('rt.reviews.tsv.gz', delimiter='\t', encoding='latin-1')
tmdb_movies = pd.read_csv('tmdb.movies.csv.gz')
tn_movie_budgets = pd.read_csv('tn.movie_budgets.csv.gz')



# Converting strings in columns to int
gross = pd.to_numeric(tn_movie_budgets.worldwide_gross.str.replace('$','').str.replace(',',''))
budget = pd.to_numeric(tn_movie_budgets.production_budget.str.replace('$','').str.replace(',',''))
domestic_gross = pd.to_numeric(tn_movie_budgets.domestic_gross.str.replace('$','').str.replace(',',''))

# Reassigning the columns in the df to be integers and making an ROI column
tn_movie_budgets['worldwide_gross'] = gross
tn_movie_budgets['production_budget'] = budget
tn_movie_budgets['domestic_gross'] = domestic_gross

# Creating ROI and it's column
roi = (gross - budget)/budget
tn_movie_budgets['roi'] = roi



# Checking out the highest ROI
roi.sort_values(ascending=False); # Highest ROI index is 5745

# What movie is the highest ROI?
tn_movie_budgets.iloc[5745]
# It's porn. With a name like Microsoft, porn might not be our forte.


# Next highest:
roi.sort_values(ascending=False) # Index 5613
tn_movie_budgets.iloc[5613]; # Mad Max. Ok. ROI of Mad Max was 497.75


# Grab the movies from the data frame between the years 2010 and 2018
# First, start get a mask to filter the df later
mask = (pd.to_datetime(tn_movie_budgets.release_date) >= '2010-01-01') &\
        (pd.to_datetime(tn_movie_budgets.release_date) < '2019-01-01')


# Filtering by date:
tn_movie_budgets_to_date = tn_movie_budgets[mask]

tn_movie_budgets_to_date.shape#.sort_values('roi', ascending=False)

# Now we want to join this data frame to another dataframe with genre info
# Setting the index of each to movies and joining on the index   
tn_movie_budgets_to_date.index = tn_movie_budgets_to_date['movie']
imdb_title_basics.index = imdb_title_basics['primary_title']

# making a the data frame that has all the info we are interested in
the_df = tn_movie_budgets_to_date.join(imdb_title_basics, how = 'inner')

# Any ROI with the value -1 is likely missing data. Writing a loop to replace this with nan
for index, element in enumerate(the_df['roi']):
    if (element < -0.9999999) & (element > -1.01):
        the_df['roi'][index] = np.nan
        
# Dropping rows with any nan values
the_df.dropna(inplace=True)

# Sort genres by median ROI
test_median = the_df.groupby('genres').roi.median().sort_values(ascending = False)

# Sample size is of concern. We want to drop any genre with less than 10 movies
# Counting the number of movies in each genre
test_count = the_df.groupby('genres').roi.count().sort_values(ascending = False)
count_df = pd.DataFrame(test_count)
median_df = pd.DataFrame(test_median)
joined_genre_df = median_df.join(count_df, how ='inner', lsuffix='median', rsuffix = 'count')


# Ok, sort by genre by count
joined_genre_df.sort_values(by='roicount', ascending=False) 

# Only keep genres that have more than 10 movies
good_genres = joined_genre_df[joined_genre_df['roicount'] >= 10]
good_genres = good_genres.sort_values(by = 'roimedian', ascending = False)

# Making a list of good genres to use
list_of_good_genres = list(good_genres.index)

# Select these genres out of our data frame
index_list = []
for index, element in enumerate(the_df['genres']):
    if element in list_of_good_genres:
        index_list.append(index)


good_sample_df = the_df.iloc[index_list,:]

# Dealing with duplicate movies
good_sample_df.index.duplicated().sum()
# There are 309 duplications


# Getting rid of duplicates
good_sample_df2 = good_sample_df.copy() # Backup copy just in case
good_sample_df2.drop_duplicates(subset='movie', keep='first',inplace = True)

# Dropping duplicate columns
good_sample_df2.drop(columns=['movie', 'primary_title', 'original_title'],inplace = True)

genre_df = good_sample_df2.copy()
movies_df = good_sample_df2.copy()# Now the dataframe is ready






## Adding dataframe coded for scatterplot roi to budget to profit comparison
#####Joe's edits#######
#similar date filter for df_movie_budgets for the roi/budget/profit scatterplot, built on same code, copy df created to avoid conflict if variable names changed
mask = (pd.to_datetime(tn_movie_budgets.release_date) >= '2010-01-01') & (pd.to_datetime(tn_movie_budgets.release_date) < '2019-01-01')
df_movie_budgets_clip = tn_movie_budgets[mask]

#df_movie_budgets_clip['worldwide_gross_numeric'] = pd.to_numeric(df_movie_budgets_clip.worldwide_gross.str.replace('$','').str.replace(',',''))
#df_movie_budgets_clip['budget_numeric'] = pd.to_numeric(df_movie_budgets_clip.production_budget.str.replace('$','').str.replace(',',''))
df_movie_budgets_clip['profit'] = df_movie_budgets_clip['worldwide_gross'] - df_movie_budgets_clip['production_budget']

df_movie_budgets_clip.rename({'id':'movie'}, inplace = True)

df_movie_budgets_clip['roi'] = df_movie_budgets_clip['profit']/df_movie_budgets_clip['production_budget']
df_movie_budgets_clip.join(imdb_name_basics, how = 'inner')
df_movie_budgets_clip.set_index('movie', inplace=True)
df_movie_budgets_clip['roi'].sort_values(ascending=False)
imdb_title_basics.set_index('primary_title', inplace = True)
## Create the_df which is where ROI/Budget/Profit pulls from
a_df = df_movie_budgets_clip.join(imdb_title_basics, how = 'inner')
a_df = a_df.loc[~a_df.index.duplicated(keep='first')]

##  Create subdataframes for ROI/Budget/Profit plot (plot dataframe is gross_roi_max_df) and then combines them into gross_roi_max_df

top_5_budget = a_df.sort_values(by = 'production_budget', ascending = False, axis = 0).iloc[:10,:]
top_5_budget['top_X_parameter'] = 'production_budget'
top_5_profit = a_df.sort_values(by = 'profit', ascending = False, axis = 0).iloc[:10,:]
top_5_profit['top_X_parameter'] = 'gross'
top_5_roi = a_df.sort_values(by = 'roi', ascending = False, axis = 0).iloc[:10,:]
top_5_roi['top_X_parameter'] = 'roi'
## Stack the dataframes
transition1_df = top_5_budget.append(top_5_profit)
gross_roi_max_df = transition1_df.append(top_5_roi)
gross_roi_max_df = gross_roi_max_df.reset_index()

## Creating indicator column for films that made money vs those that did not - will use for hue parameter in plot below
gross_roi_max_df['made_a_profit'] = 1
for row in range(0,gross_roi_max_df.shape[0]):
    if gross_roi_max_df.profit[row] > 0:
        gross_roi_max_df.made_a_profit[row] = 1
    else:
        gross_roi_max_df.made_a_profit[row] = 0
## rename column 'index' as title 
gross_roi_max_df.rename(columns = {'index':'title'}, inplace = True)

sns.set_context('poster')
ax1=sns.scatterplot(data=gross_roi_max_df, x='production_budget', y='profit', s=gross_roi_max_df['roi']*250, hue='made_a_profit')
rcParams['figure.figsize'] = (20, 14)
ax1.set_xlabel('Budget (hundreds of millions)')
ax1.set_ylabel('Profit (billions)')
plt.xticks(rotation=90) 
ax1.set_title('Return on Investment (ROI) relative to Budget and Profit', fontsize = 40)
ax1.get_legend().remove()

for line in range(0,gross_roi_max_df.shape[0]):
     ax1.text(gross_roi_max_df.production_budget[line]+0.2, gross_roi_max_df.profit[line], gross_roi_max_df.title[line], horizontalalignment='left', size='small', color='black', weight='semibold')

ax1.text(0.05, 0.95, s = 'Larger circles have larger ROI', transform=ax1.transAxes, fontsize=35, verticalalignment='top')
plt.savefig('roi_v_budget_v_profit.png')
plt.show()

# First, finding benefit
benefit = genre_df.groupby('genres').median()['roi'] 

# Now finding risk
total_genre_count = genre_df['genres'].value_counts() # Good
total_genre_count_and_one = total_genre_count + 1 #Good # Adding 1 to avoid divsion by zero 
                                                #(incase a certain genre did not lose money)
#Action,Adventure,Sci-Fi and Action,Adventure,Thriller have no failed profits. Scaling all movies up by 1.
failed_in_genre_count = genre_df[genre_df['roi']<0].groupby('genres').count()['roi'] + 1
failed_in_genre_count['Action,Adventure,Sci-Fi'] = 1
failed_in_genre_count['Action,Adventure,Thriller'] = 1

risk = np.array(failed_in_genre_count)/np.array(total_genre_count_and_one )

# now we have our Benefit-to-risk ratio
b2r = benefit/risk

# First, find benefit (median ROI)
median_roi = movies_df.groupby('genres').median()['roi'].to_frame()
median_roi.columns = ['median_roi']
# Now count the number of losers in each genre, if zero won't have a row for that genre
losers = movies_df[movies_df['roi']<0].groupby('genres').count()['roi'].to_frame()
losers.columns = ['losers']
total = movies_df.groupby('genres').count()['roi'].to_frame()
total.columns = ['total']
genres_df = median_roi.join(total, how = 'outer').join(losers, how='left').fillna(0)
# Now find the adjusted risk (losers + 1)/(total + 1) and BtoR
genres_df['adjusted_risk'] = (genres_df.losers + 1)/(genres_df.total + 1)
genres_df['BtoR'] = genres_df.median_roi/genres_df.adjusted_risk
genres_df.sort_values('BtoR', ascending=False)

# Get genres with small sample sizes
genres_to_drop = genres_df[genres_df['total']<10]


# Drop movies from genres with small samples
bigsample_movies = movies_df.copy()
bigsample_movies['title'] = bigsample_movies.index
bigsample_movies.set_index('genres', inplace=True, drop = False)
bigsample_movies.drop(genres_to_drop.index, axis= 0, inplace=True)
bigsample_movies.set_index('title', inplace=True)
# Drop genres with small samples
bigsample_genres = genres_df.copy()
bigsample_genres.drop(genres_to_drop.index, inplace=True)
#rank the genres by BtoR
ranked_genres = list(bigsample_genres.sort_values('BtoR', ascending=False).index)

# Plot the box plot
sns.set_context("poster") # Make it presentable in notebook
ax = sns.boxplot(data = bigsample_movies, y='genres', x='roi', order=ranked_genres) # makes the boxplot
ax.set(xscale='symlog') # allows for negative values on log scale
ax.set_xlabel("ROI") # label
rect = Rectangle((-5,-1), 5, 100, color='red', alpha=.3) # creates rectangle that signifies loss of money
ax.add_patch(rect) # adds the rectangle
rcParams['figure.figsize'] = 14, 14 # controls figure size
ax.set(xlim=(-1,500))
plt.xticks(ticks=[-1,0,1,10,100],labels=['-1','0','1','10','100'])
rcParams['figure.figsize'] = 14, 14

# Puts the x axis on the top and bottom
#plt.rcParams['xtick.bottom'] = True
#plt.rcParams['xtick.top'] = True

ax.set_title('ROI')
plt.show()

# Get a list of top 5 genres
top5_genres = ranked_genres[:5]
top5_genres

i_top = []
for i, element in enumerate(bigsample_movies['genres']):
    if element in top5_genres:
        i_top.append(i)

top5_df = bigsample_movies.iloc[i_top, :]


fg = sns.catplot(data = top5_df, y='genres', x='roi', order=ranked_genres[0:5])
fg.set(xscale='symlog')
#fg.set_xlabel("ROI")
#fg.set_ylabel('Top 5 Benefit-to-Risk Genres')
#rect = Rectangle((-5,-1), 5, 100, color='red', alpha=.3)
#ax.add_patch(rect)
fg.set(xlim=(-1,500))
plt.xticks(ticks=[-1,0,1,10,100],labels=['-1','0','1','10','100'])
rcParams['figure.figsize'] = 14, 14

plt.show()


# Preparing to plot roi vs budget by genre
horror_df = movies_df[movies_df['genres'] == 'Horror,Mystery,Thriller']
#Action,Adventure,Sci-Fi = AASF 
AASF_df = movies_df[movies_df['genres'] == 'Action,Adventure,Sci-Fi']
# Adventure,Animation,Comedy = AAC
AAC_df = movies_df[movies_df['genres'] == 'Adventure,Animation,Comedy']
# Comedy,Romance 
CR_df = movies_df[movies_df['genres'] == 'Comedy,Romance']
# Comedy,Drama,Romance
CDR_df = movies_df[movies_df['genres'] == 'Comedy,Drama,Romance']


plt.figure(0)
plt.scatter(x = horror_df['production_budget'], y=horror_df['roi'], color = 'k', label = 'Horror,Mystery,Thriller', alpha=0.7)
plt.scatter(x=AASF_df['production_budget'], y=AASF_df['roi'], color ='b', label = 'Action,Adventure,Sci-Fi',alpha = 0.7)
plt.scatter(x=AAC_df['production_budget'], y=AAC_df['roi'], color='orange', label = 'Adventure,Animation,Comedy', alpha=0.7)
plt.scatter(x=CR_df['production_budget'], y=CR_df['roi'], color='pink', label = 'Comedy,Romance', alpha=0.7)
plt.scatter(x=CDR_df['production_budget'], y=CDR_df['roi'], color='g', label = 'Comedy,Drama,Romance', alpha=0.7)
plt.xscale('log')
plt.title('  Top 5 Genres \n ROI vs Budget')
plt.legend(loc='best')
rcParams['figure.figsize'] = 25, 8
plt.rcParams['xtick.top'] = False
plt.xticks([1e5, 1e6,1e7,1e8],['$100,000', '$1M', '$10M', '$100M'])
plt.ylim(-10,80)
plt.xlabel('Budget Spending (USD) \n \n *Horror,Mystery,Thriller outlier at ROI = 400 removed (The Gallows)')
plt.ylabel('Return On Investment (ROI)')

























