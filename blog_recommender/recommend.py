import pandas as pd
import numpy as np


def get_title_from_index(index):
    return df[df.index == index]["Title"].values[0]


def get_index_from_title(title):
    return df[df.Title == title]["index"].values[0]


df = pd.read_csv('Medium_Clean.csv',  nrows=5000)
features = ['Title', 'Author', 'Publication', 'Author_url']
for feature in features:
    df[feature] = df[feature].fillna('')

# get precomputed similarity matrix
similarity_score = np.genfromtxt('similarity-score.csv', delimiter=',')
blog_user_read = "When is a building smart?"

# Step 6: Get index of this blog from its title
blog_index = get_index_from_title(blog_user_read)
similar_blog = list(enumerate(similarity_score[blog_index]))
# Step 7: Get a list of similar blog in descending order of similarity score
recommendations = sorted(similar_blog, key=lambda x: x[1], reverse=True)

# Step 8: Print titles of first 50 blog
i = 0
for blog in recommendations:
    title = get_title_from_index(blog[0])
    if title == '':
        continue
    else:
        print(get_title_from_index(blog[0]))
    i = i + 1
    if i > 50:
        break
