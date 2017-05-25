from __future__ import print_function
import numpy as np
import string
import sys
import keras.backend as K
import pickle
import os.path
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model,Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU,LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import RandomNormal

np.random.seed(1717)

embedding_dim = 100
thresh = 0.4
stop_words = pickle.load(open("stopwords.p","rb"))


def load_data(path, training):
	with open(path,'r') as f:
	
		tags = []
		articles = []
		tags_list = []
		
		f.readline()
		for line in f:
			if training:
				start = line.find('\"')
				end = line.find('\"',start+1)
				tag = line[start+1:end].split(' ')
				article = remove_stopwords(line[end+2:])
				
				for t in tag :
					if t not in tags_list:
						tags_list.append(t)
			   
				tags.append(tag)
			else:
				start = line.find(',')
				article = remove_stopwords(line[start+1:])
			
			articles.append(article)
			
		if training :
			assert len(tags_list) == 38,(len(tags_list))
			assert len(tags) == len(articles)

	return (tags,articles,tags_list)

def save_data(filename,Y_pred,thresh,tag_list):
	with open(filename,'w') as output:
		print ('\"id\",\"tags\"',file=output)
		Y_pred_thresh = (Y_pred > thresh).astype('int')
		for index,labels in enumerate(Y_pred_thresh):
			labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
			if len(labels) is 0:
				labels = [tag_list[i] for i,value in enumerate(Y_pred[index]) if value > 0.2 ]
				if len(labels) is 0:
					labels = [tag_list[i] for i,value in enumerate(Y_pred[index]) if value > 0.1 ]
			labels_original = ' '.join(labels)
			print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

def build_model(num_words, embedding_dim, weights, input_length, trainable):
	model = Sequential()
	model.add(Embedding(num_words,
						embedding_dim,
						weights=weights,
						input_length=input_length,
						trainable=trainable))
	#model.add(GRU(200,activation='tanh',dropout=0.2,kernel_initializer = 'glorot_uniform', bias_initializer = RandomNormal(mean = 0.0 , stddev = 0.01)))
	model.add(GRU(128,activation='tanh',dropout=0.2))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.15))
	model.add(Dense(38,activation='sigmoid'))
	model.summary()

	return model

def get_embedding_dict(path):
	embedding_dict = {}
	with open(path,'r') as f:
		for line in f:
			values = line.split(' ')
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			embedding_dict[word] = coefs
	return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
	embedding_matrix = np.zeros((num_words,embedding_dim))
	for word, i in word_index.items():
		if i < num_words:
			embedding_vector = embedding_dict.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
	return embedding_matrix

def remove_stopwords(text):
	return ' '.join([word for word in text.split() if word not in stop_words])

def main():
	###load training and testing data
	tag_list = pickle.load(open("tag_list.p", "rb"))
	(_, X_test, _) = load_data(sys.argv[1], training = False)

	###tokenizer
	tokenizer = pickle.load(open("tokenizer.p", "rb"))
	word_index = pickle.load(open("word_index.p","rb"))


	###convert word sequences to index sequences
	test_sequences = tokenizer.texts_to_sequences(X_test)

	###padding
	max_article_length = 207
	test_sequences = pad_sequences(test_sequences, maxlen = max_article_length)

	###get embedding matrix from glove
	#embedding_dict = get_embedding_dict('glove.6B.'+str(embedding_dim)+'d.txt')
	#num_words = len(word_index) + 1
	#embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

	###build model
	#model = build_model(num_words, embedding_dim, weights = [embedding_matrix], input_length = max_article_length, trainable = False)

	model = load_model("model.h5")
	model.load_weights("best_tuned.hdf5")

	Y_pred = model.predict(test_sequences)
	save_data(sys.argv[2], Y_pred, thresh = thresh, tag_list = tag_list)


if __name__ == '__main__':
	main()
