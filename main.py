# importing neccessary libraries
300
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Collection and Pre-Processing

# loading the data from the csv file to pandas dataframe
movies_data = pd.read_csv("movies.csv")

# printing the first 5 rows of the dataframe
# print(movies_data.head())

# number of rows and columns in the data frame
# print(movies_data.shape)

# selecting the relevant features for recommendation

selected_features = ['genres','keywords','overview','tagline','cast','director']
# print(selected_features)

# replacing the null valuess with null string (Pre-Processing of data)

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')      

# combining all the 6 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['overview']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']  
#print(combined_features)

# converting the text data to feature vectors

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)
# print(similarity)

# taking user input
movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

# getting the closest match
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

# taking the first value
close_match = find_close_match[0]

# getting the index of the movie from the dataframe
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# etting the list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))

# sorting the list on the basis of similarity score (highest to lowest)
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

#printing similar movies based on the user input
print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]

  #printing top 29 movies
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


