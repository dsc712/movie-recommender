from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()
X = cv.fit_transform(text)  # X => count matrix
print(cv.get_feature_names())  # words
print(X)  # prints internal representation of sparse matrix in sci kit
print(X.toarray())  # returns numpy array

similarity_score = cosine_similarity(X)
print(similarity_score)
