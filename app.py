import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import joblib

import streamlit as st

import warnings 

warnings.filterwarnings('ignore')

@st.cache
def load_and_preprocess_data():
	animes = pd.read_csv('data/animes_preprocessed.csv')
	ratings = pd.read_csv('data/ratings_preprocessed.csv')

	return animes, ratings



def get_title_from_index(anime_id):
    title = animes[animes['name'] == anime_id]
    title = list(title['name'])
    return title[0]


def get_index_from_title(title):
    anime_id = animes[animes['name'] == title]
    anime_id = list(anime_id['anime_id'])
    return anime_id[0]


def insert_ratings(ratings_dict, ratings):

	ids = []

	for anime_titles in ratings_dict.keys():
	    x = get_index_from_title(anime_titles)
	    ids.append(x)
	    
	id_list = [user_id] * len(ratings_dict)

	user_ratings = list(zip(id_list,ids, ratings_dict.values()))

	user_ratings = pd.DataFrame(user_ratings, columns=ratings.columns)

	ratings = pd.concat([ratings, user_ratings])

	return ratings, ids



def userCF_Mean(ratings):

    combined = pd.merge(animes,ratings,on='anime_id')

    anime_mat = combined.pivot_table(index='user_id',columns='name',values='rating_y').fillna(0)
    anime_mat_sparse = csr_matrix(anime_mat.values)
    cosine_sim = cosine_similarity(anime_mat_sparse)

    k = 10

    recommender_df = pd.DataFrame(cosine_sim, 
                                  columns=anime_mat.index,
                                  index=anime_mat.index)


    ## Item Rating Based Cosine Similarity
    cosine_df = pd.DataFrame(recommender_df[user_id].sort_values(ascending=False))
    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['user_id','cosine_sim']
    similar_usr = list(cosine_df['user_id'][1:k+1].values)
    similarities = list(cosine_df['cosine_sim'][1:k+1].values)

    sims_dict = dict(zip(similar_usr, similarities))

    similar_usr_df = anime_mat.T[similar_usr].fillna(0)

    for i, j in sims_dict.items():
        similar_usr_df[i] = similar_usr_df[i] * j

    similar_usr_df['mean rating'] = similar_usr_df[list(sims_dict.keys())].mean(numeric_only=True,axis=1)
    similar_usr_df.sort_values('mean rating', ascending=False,inplace = True)

    watched = list(ratings_dict.keys())

    similar_usr_df = similar_usr_df[~similar_usr_df.index.isin(watched)]
    
    titles = similar_usr_df.index
    mean_rating = list(similar_usr_df['mean rating'])
    
    recos = pd.DataFrame(columns=['name','mean rating'])
    recos['name'] = titles
    recos['mean rating'] = mean_rating
    
    recos = pd.merge(animes,recos,on='name')
    
    recos.sort_values(by='mean rating', ascending = False, inplace=True)
    recos.reset_index(drop=True, inplace=True)

    return recos.head(20)



@st.cache
def load_cosine_sim():

	cosine_sim = joblib.load('models/cosine_sim.sav')

	recommender_df = pd.DataFrame(cosine_sim, 
	                          columns=anime_titles['name'],
	                          index=anime_titles['name'])

	return recommender_df



def itemCF(recommender_df, anime_name):
	
	cosine_df = pd.DataFrame(recommender_df[anime_name])
	cosine_df.reset_index(level=0, inplace=True)
	cosine_df.columns = ['name','cosine_sim']

	cosine_df = pd.merge(animes, cosine_df, on='name')
	cosine_df.sort_values(by='cosine_sim',ascending=False,inplace=True)
	cosine_df = cosine_df.iloc[1:,]

	cosine_df.reset_index(inplace=True,drop=True)

	return cosine_df.head(20)


if __name__ == '__main__':

	st.title("Anime Recommender System")

	st.sidebar.markdown('[![Muhammad Hamza Adnan]\
                    (https://img.shields.io/badge/Author-@hamzaxd11-gray.svg?colorA=gray&colorB=dodgerblue&logo=github)]\
                    (https://github.com/hamzaxd11/anime-rec/)')


	reco_types = ['Home','Personalized Recommendations', 'Find Similar Anime']

	choice = st.sidebar.selectbox("Type of Recommendations" ,reco_types)

	st.markdown("Rate animes and get personalized recommendations or find animes similar to the ones you liked")

	animes, ratings = load_and_preprocess_data()

	#combined, pivot, anime_titles, model_knn = load_itemCF_data(animes,ratings)
	anime_titles = pd.read_csv('data/anime_titles.csv')


	recommender_df = load_cosine_sim()


	anime_list = anime_titles['name']
	anime_list = list(anime_list.sort_values())

	user_id = 99999

	rating_scale = [1,2,3,4,5,6,7,8,9,10]

	if choice == 'Home':
		st.markdown('**For Personalized Recommendations:**')

		st.text('1) Select the Personalized Recommendations option in the sidebar.')
		st.text('2) Select 5 animes youve already seen.')
		st.text('3) Rate them on a scale of 1 to 5.')
		st.text('4) Press the Submit ratings button at the bottom.')
		st.text('5) Enjoy your personalized recommendations!')

		st.markdown('**To find animes similar to the ones you liked:**')

		st.text('1) Select the Find Similar Animes option in the sidebar.')
		st.text('2) Select an anime.')
		st.text('3) Enjoy the recommendations!')

	elif choice == 'Personalized Recommendations':

		anime_1 = st.selectbox("Anime 1",anime_list)

		rating_1 = st.select_slider(
		'Anime 1 rating', rating_scale)


		anime_2 = st.selectbox("Anime 2",anime_list)

		rating_2 = st.select_slider(
		'Anime 2 rating', rating_scale)


		anime_3 = st.selectbox("Anime 3",anime_list)

		rating_3 = st.select_slider(
		'Anime 3 rating', rating_scale)


		anime_4 = st.selectbox("Anime 4",anime_list)

		rating_4 = st.select_slider(
		'Anime 4 rating', rating_scale)


		anime_5 = st.selectbox("Anime 5",anime_list)

		rating_5 = st.select_slider(
		'Anime 5 rating', rating_scale)


		user_animes = [anime_1,anime_2,anime_3,anime_4,anime_5]
		user_ratings = [rating_1,rating_2,rating_3,rating_4,rating_5]


		ratings_dict = dict(zip(user_animes, user_ratings))

		new_ratings, ids = insert_ratings(ratings_dict, ratings)



		if st.button('Submit ratings'):
			#st.write(ratings_dict)

			userCF = userCF_Mean(new_ratings)

			st.header("User based Collaborative Filtering")
			st.write('Finds the n most similar users to you and scores animes based on user similarity and takes the mean score of each anime')
			st.write(userCF)



	elif choice == 'Find Similar Anime':

		st.header('How this algorithm works:')
		st.write('People who rated an anime highly, also rated these animes similarly')

		anime_name = st.selectbox("Anime",anime_list)

		recos = itemCF(recommender_df, anime_name)

		st.write(recos)




	else:
		pass