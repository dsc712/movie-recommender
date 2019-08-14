import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


# Step 1: Read CSV File
df = pd.read_csv('movie_dataset.csv')
# print(df.columns)
# print(df.head())
# print(df.describe())

# Step 2: Select Features
features = ['keywords', 'cast', 'genres', 'director', 'production_companies', 'title']

# Step 3: Create a column in DF which combines all selected features
# clean the data
for feature in features:
	df[feature] = df[feature].fillna('')


def combine_features(row):
	return row['keywords'] + " " + row["cast"] + " " + row["genres"] + " " + row["director"] + " " + row["production_companies"] + " " + row["title"]


# Step 4: Create count matrix from this new combined column
df["combined_features"] = df.apply(combine_features, axis=1)
# axis = 1 means apply functions row wise, by default it applies column wise
cv = CountVectorizer()
X = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix
similarity_score = cosine_similarity(X)

movie_user_likes = "Iron Man"

# Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(similarity_score[movie_index]))
# Step 7: Get a list of similar movies in descending order of similarity score
recommendations = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# Step 8: Print titles of first 50 movies
i = 0
for movie in recommendations:
	print(get_title_from_index(movie[0]))
	i = i+1
	if i > 50:
		break
