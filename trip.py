#import the necessary Python libraries and the dataset we need for this task:

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("TripAdvisor_RestauarantRecommendation.csv")
print(data.head())

#select two columns from the dataset for the rest of the task (Name, Type):

data = data[["Name", "Type"]]
print(data.head())

print(data.isnull().sum())

#delete the rows containing null values:

data = data.dropna()

#type of the restaurant

feature = data["Type"].tolist()
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)


indices = pd.Series(data.index, index=data['Name']).drop_duplicates()

#set the name of the restaurant as an index so that find similar restaurants by giving the name of the restaurant as an input:

def restaurant_recommendation(name, similarity = similarity):
    index = indices[name]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    restaurantindices = [i[0] for i in similarity_scores]
    return data['Name'].iloc[restaurantindices]

print(restaurant_recommendation("Market Grill"))
