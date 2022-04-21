#!/usr/bin/env python
# coding: utf-8

# # Data Translation Challenge: Group 1
# ### Source: https://www.codementor.io/spark/tutorial/building-a-recommender-with-apache-spark-python-example-app-part1

# #### Create a SparkContext configured for local mode

# In[1]:


import pyspark

sc = pyspark.SparkContext("local[*]")


# #### File download
# Small: 100,000 ratings and 2,488 tag applications applied to 8,570 movies by 706 users. Last updated 4/2015.   
# Full: 21,000,000 ratings and 470,000 tag applications applied to 27,000 movies by 230,000 users. Last updated 4/2015.

# In[ ]:


full_dataset_url = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"


# #### Download location(s)

# In[2]:


import os

datasets_path = os.path.join("/home/jovyan", "work")
full_dataset_path = os.path.join(datasets_path, "ml-latest.zip")


# #### Getting file(s)

# In[ ]:


import urllib.request

full_f = urllib.request.urlretrieve(full_dataset_url, full_dataset_path)


# #### Extracting file(s)

# In[ ]:


import zipfile

with zipfile.ZipFile(full_dataset_path, "r") as z:
    z.extractall(datasets_path)


# ## Loading and parsing datasets
# We can read in each of the files and create an RDD consisting of parsed lines. 
# 

# ### ratings.csv

# In[3]:


# Load the full ratings dataset file
complete_ratings_file = os.path.join(datasets_path, "ml-latest", "ratings.csv")
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse to create the tuple.
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line != complete_ratings_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), 
                                                           int(tokens[1]), 
                                                           float(tokens[2]))).cache()

print (f"There are {complete_ratings_data.count()} recommendations in the complete dataset")
complete_ratings_data.take(3)


# ### movies.csv

# In[4]:


# Load the full movies dataset file
complete_movies_file = os.path.join(datasets_path, "ml-latest", "movies.csv")
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse to create the tuple
complete_movies_data = complete_movies_raw_data.filter(lambda line: line != complete_movies_raw_data_header)    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))
print (f"There are {complete_movies_titles.count()} movies in the complete dataset")
complete_movies_data.take(3)


# #### Selecting ALS parameters using the full dataset
# In order to determine the best ALS parameters, we will use the full dataset. We need first to split it into train, validation, and test datasets.

# In[5]:


# source uses seed=0L, which is the previous version of python (2.x)
# 0L should be written as 0 from now on
training_RDD, validation_RDD, test_RDD = complete_ratings_data.randomSplit([6.0, 2.0, 2.0], seed=0)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


# #### Training phase

# In[7]:


# Hyperparameter tuning

from pyspark.mllib.recommendation import ALS
import math

seed = 5
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float("inf")
best_rank = -1
best_iteration = -1

for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    errors[err] = error
    err += 1
    print (f"For rank {rank} the RMSE is {error}")
    if error < min_error:
        min_error = error
        best_rank = rank

print (f"The best model was trained with rank {best_rank}")


# In[8]:


# Split it into training and test datasets.
training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)

complete_model = ALS.train(training_RDD, best_rank, seed=seed,                            iterations=iterations, lambda_=regularization_parameter)


# In[10]:


# On to the test set.
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

print (f"For testing data the RMSE is {error}")


# In[11]:


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)

movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# ## Users with Own Ratings

# In[13]:


new_user_ratings_1 = [
     (0,55247,2), # Into the Wild (2007) - Action|Adventure|Drama
     (0,55245,1), # - Good Luck Chuck (2007) - Comedy|Romance
     (0,56757,1), # Sweeney Todd: The Demon Barber of Fleet Street (2007) - Drama|Horror|Musical|Thriller
     (0,52973,3), # Knocked Up (2007) - Comedy|Drama|Romance 
     (0,122900,5), # Ant-Man (2015) - Action|Adventure|Sci-Fi
     (0,122918,5), # Guardians of the Galaxy 2 (2017) - Action|Adventure|Sci-Fi 
     (0,22924,5), # X-Men: Apocalypse (2016) - Action|Adventure|Fantasy|Sci-Fi
     (0,1197,5), # Princess Bride, The (1987) - Action|Adventure|Comedy|Fantasy|Romance
     (0,1198,4), # Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) - Action|Adventure
     (0,4306,5) # Shrek (2001) - Adventure|Animation|Children|Comedy|Fantasy|RomanceShrek (2001) - Adventure|Animation|Children|Comedy|Fantasy|Romance
]

new_user_ratings_2 = [
    (0,8694,5), # Black Panther (2017)
    (0,6733,4), # Speed Racer (2008)
    (0,8029,4), # Man with the Iron Fists, The (2012)
    (0,9354,2), # The Infiltrator (2016)
    (0,6907,3), # Twilight (2008)
    (0,6905,1), # Bolt (2008)
    (0,1754,1), # Heart Condition (1990)
    (0,8026,5), # Flight (2012)
    (0,4942,3), # Man on Fire (2004)
    (0,4946,2) # Mean Girls (2004)
]


# ### User 1 - Scenario 1

# In[25]:


# Train the user 1 specific model.

new_user_ratings_RDD_1 = sc.parallelize(new_user_ratings_1)
complete_data_with_new_ratings_RDD_1 = complete_ratings_data.union(new_user_ratings_RDD_1)
new_ratings_model_1 = ALS.train(complete_data_with_new_ratings_RDD_1, best_rank, seed=seed,
                                iterations=iterations, lambda_=regularization_parameter)


# In[26]:


new_user_ID_1 = 0

# Remove rated movies.
new_user_ratings_ids_1 = map(lambda x: x[1], new_user_ratings_1)
new_user_unrated_movies_RDD_1 = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids_1).map(lambda x: (new_user_ID_1, x[0])))

# Get predicted recommendations.
new_user_recommendations_RDD_1 = new_ratings_model_1.predictAll(new_user_unrated_movies_RDD_1)

# Join datasets for information.
new_user_recommendations_rating_RDD_1 = new_user_recommendations_RDD_1.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD_1 =     new_user_recommendations_rating_RDD_1.join(complete_movies_titles).join(movie_rating_counts_RDD)

# Parse results.
new_user_recommendations_rating_title_and_count_RDD_1 =     new_user_recommendations_rating_title_and_count_RDD_1.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

# Filter for movies with, at least, 25 ratings.
top_movies_1_25 = new_user_recommendations_rating_title_and_count_RDD_1.filter(lambda r: r[2] >= 25).takeOrdered(15, key=lambda x: -x[1])

print ("TOP recommended movies user 1 (with more than 25 reviews):\n{}".format("\n".join(map(str, top_movies_1_25))))


# ### User 1 - Scenario 2

# In[28]:


# Filter for movies with, at least, 100 ratings.
top_movies_1_100 = new_user_recommendations_rating_title_and_count_RDD_1.filter(lambda r: r[2] >= 100).takeOrdered(15, key=lambda x: -x[1])

print ("TOP recommended movies user 1 (with more than 100 reviews):\n{}".format("\n".join(map(str, top_movies_1_100))))


# ### User 2 - Scenario 1

# In[29]:


# Train the user 2 specific model.

new_user_ratings_RDD_2 = sc.parallelize(new_user_ratings_2)
complete_data_with_new_ratings_RDD_2 = complete_ratings_data.union(new_user_ratings_RDD_2)
new_ratings_model_2 = ALS.train(complete_data_with_new_ratings_RDD_2, best_rank, seed=seed,
                                iterations=iterations, lambda_=regularization_parameter)


# In[30]:


new_user_ID_2 = 0

# Remove rated movies
new_user_ratings_ids_2 = map(lambda x: x[1], new_user_ratings_2)
new_user_unrated_movies_RDD_2 = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids_2).map(lambda x: (new_user_ID_2, x[0])))

# Get predicted recommendations
new_user_recommendations_RDD_2 = new_ratings_model_2.predictAll(new_user_unrated_movies_RDD_2)

# Join datasets for information
new_user_recommendations_rating_RDD_2 = new_user_recommendations_RDD_2.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD_2 =     new_user_recommendations_rating_RDD_2.join(complete_movies_titles).join(movie_rating_counts_RDD)

# Parse results
new_user_recommendations_rating_title_and_count_RDD_2 =     new_user_recommendations_rating_title_and_count_RDD_2.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

# Filter for at least 25 ratings
top_movies_2_25 = new_user_recommendations_rating_title_and_count_RDD_2.filter(lambda r: r[2] >= 25).takeOrdered(15, key=lambda x: -x[1])

print ("TOP recommended movies user 2 (with more than 25 reviews):\n{}".format("\n".join(map(str, top_movies_2_25))))


# ### User 2 - Scenario 2

# In[31]:


# Filter for at least 100 ratings
top_movies_2_100 = new_user_recommendations_rating_title_and_count_RDD_2.filter(lambda r: r[2] >= 100).takeOrdered(15, key=lambda x: -x[1])

print ("TOP recommended movies user 2 (with more than 100 reviews):\n{}".format("\n".join(map(str, top_movies_2_100))))


# In[ ]:




