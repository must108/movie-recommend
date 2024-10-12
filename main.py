import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# function to clean title with regex
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)


# search the dataset for similar titles
def search(title):
    title = title
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results


# returns the score, title, and genre columns of the most similar movies
def find_similar_movies(movie_id):
    # find all the users who liked the movie at movie_id
    # then get movies which these users also liked and
    # get percentage of users who liked similar movies
    similar_users = ratings[(ratings["movieId"] == movie_id) & 
                            (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) &
                                (ratings["rating"] > 4)]["movieId"]
    
    # get only the top 10% of users who liked a movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]

    # how common a rating is among all users (so if everyone likes a movie,
    # then it might not be the best to recommend for a specific niche)
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) &
                        (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(
                        all_users["userId"].unique())
    
    # create recommendation score
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis = 1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    # return similar movies, with only the score, title, and genres columns
    return rec_percentages.head(10).merge(
        movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    

# load movies and rating data
movies = pd.read_csv("ml-25m/movies.csv")
ratings = pd.read_csv("ml-25m/ratings.csv")

# cleans all movie titles
movies["clean_title"] = movies["title"].apply(clean_title)

# ngram range allows you to select two words to be important

# terms in a phrase are split into a tfid matrix, which a column for
# each unique word or phrase
# inverse document frequency is then performed, which finds the most important
# words/vals in a dataset

# numbers are then assigned to each val in the matrix, to compare with
# other phrases
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

inp_title = input("Enter title of the movie: ")
results = search(inp_title)
movie_id = results.iloc[0]["movieId"]
suggestions = find_similar_movies(movie_id)

if suggestions.empty:
    print("Title error! Tweak your movie title a bit! (Adding the year may also help!)")
else:
    print(suggestions)





