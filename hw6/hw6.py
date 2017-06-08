import numpy as np
import pandas as pd
import csv
import sys
from keras.models import Sequential
from keras.layers import Input, Activation ,Embedding, Reshape, Merge, Dropout, Dense, Flatten
from keras.layers.merge import Dot, Add, Concatenate
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adamax
import keras

k = 128
seed = 1717

def myModel(users_num, movies_num):
	P = Sequential()
	P.add(Embedding(users_num+1, k, input_length = 1))
	P.add(Reshape((k,)))

	Q = Sequential()
	Q.add(Embedding(movies_num+1, k, input_length = 1))
	Q.add(Reshape((k,)))
	
	model = Sequential()
	merged = Merge([P,Q], mode = 'dot', dot_axes = 1)
	model.add(merged)
	#model.add(Activation('relu'))
	model.summary()

	return model

def MFModel(users_num, movies_num, latent_dim = 666):
	user_input = Input(shape = [1])
	item_input = Input(shape= [1])
	user_vec = Embedding(users_num + 1, latent_dim)(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(users_num + 1, latent_dim)(item_input)
	item_vec = Flatten()(item_vec)
	user_bias = Embedding(users_num + 1, 1, embeddings_initializer = "zeros")(user_input)
	user_bias = Flatten()(user_bias)
	item_bias = Embedding(movies_num + 1, 1, embeddings_initializer = "zeros")(item_input)
	item_bias = Flatten()(item_bias)
	r_hat = Dot(axes = 1)([user_bias, item_bias])
	r_hat = Add()([r_hat, user_bias, item_bias])
	model = keras.models.Model([user_input, item_input], r_hat)

	return model

def nnModel(users_num, movies_num, latent_dim = 128):
	user_input = Input(shape = [1])
	item_input = Input(shape= [1])
	user_vec = Embedding(users_num + 1, latent_dim)(user_input)
	user_vec = Flatten()(user_vec)
	item_vec = Embedding(users_num + 1, latent_dim)(item_input)
	item_vec = Flatten()(item_vec)
	merge_vec = Concatenate()([user_vec, item_vec])
	hidden = Dense(150, activation = 'relu')(merge_vec)
	hidden = Dense(50, activation = 'relu')(hidden)
	output = Dense(1)(hidden)
	model = keras.models.Model([user_input, item_input], output)

	return model

def load_data(train,users,movies):

	def shuffle(data):
		shuffled = data.sample(frac = 1., random_state = seed)
		user = shuffled['UserID'].values
		movie = shuffled['MovieID'].values
		rating = shuffled['Rating'].values

		difference = rating.max() - rating.min()
		mean = rating.mean()

		rating = (rating - mean) / difference

		return (user,movie,rating),difference,mean

	data = pd.read_csv(train, sep = ',', usecols = ['TrainDataID', 'UserID', 'MovieID', 'Rating'])
	max_userid = data['UserID'].drop_duplicates().max()
	max_movieid = data['MovieID'].drop_duplicates().max()

	(user, movie, rating),difference,mean = shuffle(data)


	#user = []
	#user_index = []
	#with open(users,'rb') as users_data:
	#	row = csv.reader(users_data, delimiter = ':')
	#	row_counter = 0
	#	for r in row:
	#		if row_counter is not 0:
	#			user.append([])
	#			user[row_counter - 1].append(item for item in r if item != '')
	#			user_index.append(r[0])
	#		row_counter = row_counter + 1

	#users_num = len(user)
	#user = np.asarray(user)

	#movie = []
	#movie_index = []
	#with open(movies, 'rb') as movies_data:
	#	row = csv.reader(movies_data, delimiter = ':')
	#	row_counter = 0
	#	for r in row:
	#		if row_counter is not 0:
	#			movie.append([])
	#			movie[row_counter - 1].append(item for item in r if item != '')
	#			movie_index.append(r[0])
	#		row_counter = row_counter + 1

	#movies_num = len(movie)
	#movie = np.asarray(movie)

	#with open('train.p','r') as train_p:
	#	data = pickle.load(train_p)

	#data = np.zeros((users_num, movies_num))
	#with open(train,'rb') as training_data:	
	#	row = csv.reader(training_data, delimiter = ',')
	#	skip = True
	#	previous = '796'
	#	row_counter = 0
	#	for r in row:
	#		if not skip:
	#			if previous != r[1]:
	#				row_counter = row_counter + 1
	#			index = movie_index.index(r[2])
	#			data[row_counter][index] = int(r[3])
	#			previous = r[1]
	#		skip = False

	#with open('train.p','w') as train_p:
	#	pickle.dump(data,train_p)

	return (user,movie,rating,max_userid,max_movieid),(difference,mean)

def save_result(filepath,test, model,difference,mean):
	def rate(model, user_id, movie_id):
		user = np.asarray([user_id],dtype = np.int64)
		movie = np.asarray([movie_id],dtype = np.int64)
		return model.predict([user,movie])[0][0]

	with open(test,'rb') as test_data:
		row = csv.reader(test_data, delimiter =',')
		with open(filepath, 'wb') as result:
			writer = csv.writer(result, delimiter = ',')
			writer.writerow(["TestDataID","Rating"])
			skip = True
			for r in row:
				if not skip:
					rating = rate(model,r[1],r[2]) * difference + mean
					if rating < 0.0:
						writer.writerow([r[0],float(0)])
					elif rating > 5.0:
						writer.writerow([r[0], float(5)])
					else:
						writer.writerow([r[0],rating])
				skip = False

def main():

	#(user, movie, rating, max_userid, max_movieid),(difference,mean) = load_data("train.csv", "users.csv", "movies.csv")
	#max_userid = 6040, max_movieid = 3952
	#users_num = max_userid
	#movies_num = max_movieid
	users_num = 6040
	movies_num = 3952

	difference = 4
	mean = 3.58171208604

	model = MFModel(users_num, movies_num,latent_dim = 128)
	#model.compile(loss = 'mse', optimizer = 'adamax')
	#earlystopping = EarlyStopping('val_loss', patience = 2)
	#checkpoint = ModelCheckpoint(filepath = 'normalizedMFbest.hdf5', verbose = 1, save_weights_only = True, save_best_only=True)
	#history = model.fit([user,movie], rating, epochs = 50, validation_split = 0.1, verbose = 1, callbacks = [earlystopping, checkpoint])

	#model.load_weights("normalizedMFbest.hdf5")

	#opt = Adamax(lr = 0.00002)
	#model.compile(loss = 'mse', optimizer = opt)
	#earlystopping = EarlyStopping('val_loss', patience = 2)
	#checkpoint = ModelCheckpoint(filepath = 'normalizedMFbest_tuned.hdf5', verbose = 1, save_weights_only = True, save_best_only=True)
	#history = model.fit([user,movie], rating, epochs = 50, validation_split = 0.1, verbose = 1, callbacks = [earlystopping, checkpoint])

	model.load_weights("normalizedMFbest_tuned.hdf5")
	save_result(sys.argv[2],sys.argv[1]+'test.csv', model,difference, mean)


if __name__ == '__main__':
	main()