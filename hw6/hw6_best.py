import numpy as np
import pandas as pd
import csv
import sys
from keras.models import Sequential
from keras.layers import Activation ,Embedding, Reshape, Merge, Dropout, Dense
from keras.layers.merge import Dot
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

def load_data(train,users,movies):

	#def shuffle(data):
	#	shuffled = data.sample(frac = 1., random_state = seed)
	#	user = shuffled['UserID'].values
	#	movie = shuffled['MovieID'].values
	#	rating = shuffled['Rating'].values

	#	return (user,movie,rating)

	#data = pd.read_csv(train, sep = ',', usecols = ['TrainDataID', 'UserID', 'MovieID', 'Rating'])
	#max_userid = data['UserID'].drop_duplicates().max()
	#max_movieid = data['MovieID'].drop_duplicates().max()

	#(user, movie, rating) = shuffle(data)


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

	return (user,movie,rating,max_userid,max_movieid)

def save_result(filepath,test, model):
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
					rating = rate(model,r[1],r[2])
					if rating < 0.0:
						writer.writerow([r[0],float(0)])
					elif rating > 5.0:
						writer.writerow([r[0], float(5)])
					else:
						writer.writerow([r[0],rating])
				skip = False

def main():

	#(user, movie, rating, max_userid, max_movieid) = load_data("train.csv", "users.csv", "movies.csv")
	#max_userid = 6040, max_movieid = 3952
	#users_num = max_userid
	#movies_num = max_movieid
	users_num = 6040
	movies_num = 3952

	model = myModel(users_num, movies_num)
	#model.compile(loss = 'mse', optimizer = 'adamax')
	#earlystopping = EarlyStopping('val_loss', patience = 2)
	#checkpoint = ModelCheckpoint(filepath = '/home/jennyz1105/best.hdf5', verbose = 1, save_weights_only = True, save_best_only=True)
	#history = model.fit([user,movie], rating, epochs = 12, validation_split = 0.1, verbose = 1, callbacks = [earlystopping, checkpoint])

	#model.load_weights("best.hdf5")

	#opt = Adamax(lr = 0.00002)
	#model.compile(loss = 'mse', optimizer = opt)
	#earlystopping = EarlyStopping('val_loss', patience = 2)
	#checkpoint = ModelCheckpoint(filepath = 'best_tuned.hdf5', verbose = 1, save_weights_only = True, save_best_only=True)
	#history = model.fit([user,movie], rating, epochs = 50, validation_split = 0.1, verbose = 1, callbacks = [earlystopping, checkpoint])

	model.load_weights("best_tuned.hdf5")
	save_result(sys.argv[2],sys.argv[1]+'/test.csv', model)


if __name__ == '__main__':
	main()