import numpy
import sys

#check to see if there are arguments
if(len(sys.argv) is 3):
	matrixA = numpy.genfromtxt(sys.argv[1], dtype = 'i', delimiter = ',')
	matrixB = numpy.genfromtxt(sys.argv[2], dtype = 'i', delimiter = ',')
else:
	matrixA = numpy.genfromtxt("matrixA.txt", dtype = 'i', delimiter = ',')
	matrixB = numpy.genfromtxt("matrixB.txt", dtype = 'i', delimiter = ',')

#do the multiplication and iterate over the result to obtain all numbers
matrixC = numpy.matmul(matrixA, matrixB)
myList = list()
columns = len(matrixC)
if len(matrixC.shape) is 1:
	while columns > 0:
		myList.append(matrixC[columns - 1])
		columns = columns - 1
elif len(matrixC.shape) > 1:
	while columns > 0:
		rows = len(matrixC[columns - 1])
		while rows > 0:
			myList.append(matrixC[columns-1][rows-1])
			rows = rows - 1
		columns = columns - 1

#sort the list of numbers and save them to destination
myList.sort()
file = open('ans_one.txt','w')
for number in myList:
	file.write(str(number)+'\n')
file.close()
