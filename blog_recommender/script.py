from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_title_from_index(index):
    return df[df.index == index]["Title"].values[0]


def get_index_from_title(title):
    return df[df.Title == title]["index"].values[0]


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

blog_user_read = "Futures of AI, Friendly AI?"

# Step 6: Get index of this blog from its title
blog_index = get_index_from_title(blog_user_read)
similar_blog = list(enumerate(similarity_score[blog_index]))
# Step 7: Get a list of similar blog in descending order of similarity score
recommendations = sorted(similar_blog, key=lambda x: x[1], reverse=True)

# Step 8: Print titles of first 50 blog
i = 0
for blog in recommendations:
    print(get_title_from_index(blog[0]))
    i = i + 1
    if i > 50:
        break
