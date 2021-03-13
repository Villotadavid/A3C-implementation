import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
import numpy as np
import re
import sys

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(file):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)  



def main():
	p_obj=[0,0]
	general='./Dataset'
	experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(experiments)
	n=0
	maxi=0
	k= 50
	print experiments
	for exp in range (0,number_exp):

	   if experiments[exp]=='./Desktop/Dataset/OLD':
		pass
	   else:
		k=k+1
		#print experiments[exp]+'/images/*'
		frames=sorted(glob.glob(experiments[exp]+'/images/*'))
		#print frames
		number_files = len(frames)
		n=0
		maxi=0
		foldername='Exp'+str(k)
		path='./DATA/'+foldername
		print path+'     '+str(number_files)
		while os.path.isdir(path):
			k=k+1
			foldername='Exp'+str(k)
			path='./DATA/'+foldername

		try:
    			os.mkdir(path)
			os.mkdir(path+'/images')
			for fr in range (0,number_files):
				#print frames[n]
				filex = open(frames[n], 'r')
				stereo,a =load_pfm(filex)
				if stereo.max()>maxi:
					maxi=stereo.max()
				else:
					maxi=maxi
				n=n+1	
			n=0
			for fr in range (0,number_files):
				filex = open(frames[n], 'r')
				stereo,a =load_pfm(filex)
				#print maxi
				stereo=stereo*255/(maxi)
		
				cv2.imshow('img',stereo)
				cv2.waitKey(40)
				frname='frame_{0:05d}'.format(fr)+'.png'
				cv2.imwrite(path+'/images/'+frname, stereo)
				#f = open('./Dataset/images2'+'/'+frname, "w")
				#f.write(stereo)
				n=n+1		
		except OSError:
    			print ("Creation of the directory %s failed" % path)
			

		


	#print 'numero de fotos es', n
if __name__ == "__main__":
    main()







