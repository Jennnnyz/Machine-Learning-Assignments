import numpy
from PIL import Image
import sys

#check to see if there are arguments and convert files to arrays
if(len(sys.argv) is 3):
	original = numpy.array(Image.open(sys.argv[1]).convert('RGBA'))
	modified = numpy.array(Image.open(sys.argv[2]).convert('RGBA'))
else:
	original = numpy.array(Image.open('lena.png').convert('RGBA'))
	modified = numpy.array(Image.open('lena_modified.png').convert('RGBA'))

#calculate the difference
difference = numpy.subtract(original, modified)
dimension = difference.shape
rows = dimension[0]
while rows > 0:
	columns = dimension[1]
	while columns > 0:
		#if there is a difference, use the RGBA in the modified image
		if difference[rows - 1][columns - 1].any():
			difference[rows - 1][columns - 1] = modified[rows - 1][columns - 1]
		columns = columns - 1
	rows = rows - 1

#save the image
img = Image.fromarray(difference,'RGBA')
img.save('ans_two.png')