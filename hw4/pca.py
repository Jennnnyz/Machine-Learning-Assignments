import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_data(folder):
	#first 13 subjects
	alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
	data = []

	for i in range(10):
		for j in range(10):
				name = folder + alphabets[i] + '0' + str(j) + ".bmp"
				img = Image.open(name)
				data.append(np.array(img))
	data =np.asarray(data).reshape(100, 64*64)


	return np.asarray(data)

def rmse(real, estimated):
	return np.sqrt(((real - estimated)**2).mean())*100

def PCA(data, k):
	mean = np.mean(data, axis = 0)
	#draw average face
	#draw(mean,1,1,1)
	data = data - mean
	u,s,v = np.linalg.svd(data)
	#draw top 9 eigenfaces
	#draw(v,9,3,3)

	#project 100 faces on top 5 eigenfaces
	new_data = []
	for j in range(100):
	#	for i in range(5):
		new_data.append(np.dot(np.dot(data[j],v[:k,:].T), v[:k,:]))
	new_data = np.asarray(new_data) + mean
	print(rmse((data+mean)/256, new_data/256))


	#draw(new_data, 100, 10, 10)


def draw(images, iterations, m, n):
	plt.figure()
	for i in range(iterations):
		plt.subplot(m, n, i+1)
		plt.imshow(images[i].reshape(64,64),cmap = 'gray')
		plt.axis('off')
		plt.draw()
	plt.show()





def main():
	data = load_data("face/")
	PCA(data, 59)

if __name__ == "__main__":
    main()