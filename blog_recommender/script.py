from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np


# Step 1: Read CSV File
df = pd.read_csv('Medium_Clean.csv',  nrows=5000)

# Step 2: Select Features
features = ['Title', 'Author', 'Publication', 'Author_url']

# Step 3: Create a column in DF which combines all selected features
# clean the data
for feature in features:
    df[feature] = df[feature].fillna('')


def combine_features(row):
    return row["Title"] + " " + row["Author"]


# Step 4: Create count matrix from this new combined column
df["combined_features"] = df.apply(combine_features, axis=1)
# axis = 1 means apply functions row wise, by default it applies column wise
cv = CountVectorizer()
X = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix
similarity_score = cosine_similarity(X)
# save similarity matrix to csv
np.savetxt("similarity-score.csv", similarity_score, delimiter=",")
