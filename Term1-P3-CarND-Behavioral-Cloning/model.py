


import numpy as np
import cv2
import os
import csv

import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Helper function
def show_img(img, title_text = 'Image'):
	cv2.imshow(title_text, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Helper function
def save_img(img, img_name, location):
	cv2.imwrite(location + '/' + img_name, img)


# Helper function
def show_histogram(bins, hist, title_text = ''):
	width = 0.9 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align = 'center', width = width)
	plt.title(title_text)
	plt.xlabel('Angle, rescaled to [-1, 1]')
	plt.ylabel('# of angles')
	plt.show()




# Flattens the distribution of steering angles (i.e removes huge "spikes" in the histogram).
# Decrease flatten_scale: discard more data and vice versa
# Returns a list containing all indices to be deleted to flatten the distribution
def flatten_data_distribution(hist, bins, steering_angles, flatten_scale = 1):
	# Calculate probabilities for keeping any given angle
	avg_angles_per_bin = len(steering_angles) / (len(bins) - 1)
	keep = np.ones(steering_angles.shape)
	for i in range(len(bins) - 1):
	    if hist[i] >  (flatten_scale * avg_angles_per_bin):
	        keep[i] = (flatten_scale * avg_angles_per_bin) / (hist[i])
	indices_to_delete = []

	for i in range(len(steering_angles)):
	    for j in range(len(bins) - 1):
	    	if np.random.rand() > keep[j]:
	    		if bins[j] < steering_angles[i] <= bins[j + 1]:
	    			indices_to_delete.append(i)
	return indices_to_delete

# Image pre-processing
# The pre processing is done with the nVidia network architecture (suggested by Udacity) in mind
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def pre_process(img):

	# Step 1: crop
	# As suggested by Udacity, parts of the image are not helpfull when training the model
	# Cropping in Keras is possible, but I prefer to do it here
	pre_processed_img = img[70 : 130, :, :]

	# Step 2: Resize 
	# Make sure that the image dimensions matches the network architecture
	# Input size: 3x66x200, (nVidia paper, Figure 4)
	pre_processed_img = cv2.resize(pre_processed_img,(200, 66), interpolation = cv2.INTER_AREA)

	# Step 3: Convert to YUV color space 
	# Suggested in the Nvidia paper, (Section 4: Network Architecture)
	pre_processed_img = cv2.cvtColor(pre_processed_img, cv2.COLOR_BGR2YUV)
	return pre_processed_img



# Suggested by nVidia paper (Section 5.2: Augmentation)
# Helpful: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#
def apply_random_shift(img, max_shift = 1/9):
	shifted_img = img.astype(float) 


	n_rows, n_cols, _ = shifted_img.shape

	n_rows_scaled = 0.4 * n_rows

	row_shift = np.random.randint(- (n_rows * max_shift) , (n_rows * max_shift))

	pts1 = np.float32([[0, n_rows_scaled], [n_cols, n_rows_scaled], [0 , n_rows] , [n_cols, n_rows]])
	pts2 = np.float32([[0, n_rows_scaled + row_shift], [n_cols, n_rows_scaled + row_shift], [0, n_rows] , [n_cols, n_rows]])

	M = cv2.getPerspectiveTransform(pts1, pts2) 
	shifted_img = cv2.warpPerspective(shifted_img, M , (n_cols, n_rows), borderMode = cv2.BORDER_REPLICATE)

	return shifted_img.astype(np.uint8)

# Suggested by nVidia paper (Section 5.2: Augmentation)
# Helpful: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#
def apply_random_rotation(img, max_rot = 5):

	rotated_img = img.astype(float) 

	n_rows, n_cols, _ = img.shape

	img_center = (n_rows / 2, n_cols / 2)

	random_angle = np.random.randint(-max_rot, max_rot)

	R  = cv2.getRotationMatrix2D(img_center, random_angle, 1.0)

	rotated_img = cv2.warpAffine(img, R, (n_cols, n_rows))

	return rotated_img.astype(np.uint8)

# From http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#
def get_nVidia_model():
	model = Sequential()

	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (66, 200, 3)))

	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu', border_mode='valid'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu',  border_mode='valid'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu',  border_mode='valid'))
	
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
	model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))

	model.add(Flatten())

	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	return model

def generate_data(img_paths, steering_angles, batch_size, augment_img = True):

	img_paths, steering_angles = shuffle(img_paths, steering_angles)

	inputs, outputs =  ([], [])

	while True:
		for i in range(len(steering_angles)):
			img = cv2.imread(img_paths[i])
			steering_angle = steering_angles[i]
			img = pre_process(img)
			if augment_img:
				img = apply_random_shift(img)
				img = apply_random_rotation(img)
			inputs.append(img)
			outputs.append(steering_angle)
			inputs.append(cv2.flip(img, 1))
			outputs.append(-1.0 * steering_angle)
			if len(inputs) == batch_size:
				yield (np.array(inputs), np.array(outputs))
				inputs, outputs =  ([], [])
				img_paths, steering_angles = shuffle(img_paths, steering_angles)





def main():
	#tt = "Test track", ct = "Challenge Track"
	use_udacity_data = True
	use_my_data_tt = True
	use_my_data_ct = True

	SPEED_THRESHOLD = 0.1 # Frames with speed < this threshold will be discarded
	
	# This angle is added to the left and right images. 
	# Meant to help simulate recovery (car drives toward the track egde, but manages to get back to the center)
	STEERING_ANGLE_OFFSET = 0.4 

	N_BINS = 25 # Number of bins in steering_angle histogram

	train = True

	debug = False

	# Path to save some images used for the writeup
	img_writeup_path = '/home/oystein/Projects/Udacity/CarND-Self-Driving-Car/CarND-Behavioral-Cloning-P3/images'

	data_path = '/media/oystein/Storage/ubuntu/udacity/P3-data/'
	folders = ['my-data-tt', 'my-data-ct', 'udacity-data']
	data_to_use = [use_my_data_tt, use_my_data_ct,  use_udacity_data]

	csv_paths = []
	img_base_paths = []

	# Construct the driving_long.csv paths and prepare the "base" for the image paths
	for i in range(len(folders)):
		if data_to_use[i]:
			csv_paths.append(data_path + folders[i] + '/driving_log.csv')
			img_base_paths.append(data_path + folders[i] + '/IMG/')


	img_paths = []
	steering_angles = []
	# Based on the csv paths, construct the full path to each left, center and right image 
	for i in range(len(csv_paths)):

		with open(csv_paths[i], newline = '') as f:
			driving_log = list(csv.reader(f, skipinitialspace = True, delimiter = ',', quoting = csv.QUOTE_NONE))

			# driving_log columns: center, left, right, steering, throttle, brake, speed
			for line in driving_log[1:]:
				if float(line[6]) > SPEED_THRESHOLD:

					# Center
					img_paths.append(img_base_paths[i] + os.path.basename(line[0]))
					steering_angles.append(float(line[3]))

					# Left
					img_paths.append(img_base_paths[i] + os.path.basename(line[1]))
					steering_angles.append(float(line[3]) + STEERING_ANGLE_OFFSET)

					# Right 
					img_paths.append(img_base_paths[i] + os.path.basename(line[2]))
					steering_angles.append(float(line[3]) - STEERING_ANGLE_OFFSET)

	
	img_paths = np.array(img_paths)
	steering_angles = np.array(steering_angles)

	
	
	if debug == True:
		sample_img = cv2.imread(img_paths[0])
		save_img(sample_img, 'before_pre_processing.png', img_writeup_path)
		pre_processed_img = pre_process(sample_img)
		save_img(pre_processed_img, 'after_pre_processing.png', img_writeup_path)
		rotated_img = apply_random_rotation(sample_img);
		save_img(rotated_img, 'rotated_img.png', img_writeup_path)
		shifted_img = apply_random_shift(sample_img)
		save_img(shifted_img, 'shifted_img.png', img_writeup_path)

		

	
	hist, bins = np.histogram(steering_angles, N_BINS)
	if debug == True:
		show_histogram(bins, hist, 'Steeering angle distribution BEFORE flattening')



	
	indices_to_delete = flatten_data_distribution(hist, bins, steering_angles, flatten_scale = 0.2)
	img_paths = np.delete(img_paths, indices_to_delete)
	steering_angles = np.delete(steering_angles, indices_to_delete)
	hist, bins = np.histogram(steering_angles, N_BINS)
	if debug == True:
		show_histogram(bins, hist, 'Steeering angle distribution AFTER flattening')


	BATCH_SIZE = 64
	if train:
		img_train_paths, img_test_paths, steering_angles_train, steering_angles_test = train_test_split(img_paths, steering_angles, test_size = 0.2)

		model = get_nVidia_model()
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse')

		training_data_gen = generate_data(img_train_paths, steering_angles_train, BATCH_SIZE, augment_img = False)

		validation_data_gen = generate_data(img_test_paths, steering_angles_test, BATCH_SIZE, augment_img = True)

		history_object = model.fit_generator(training_data_gen, validation_data = validation_data_gen, nb_val_samples = 2500, samples_per_epoch = 20000, nb_epoch = 2, verbose = 1)
		model.save('./model.h5')

		print(history_object.history.keys())
		print('Loss')
		print(history_object.history['loss'])
		print('Validation Loss')
		print(history_object.history['val_loss'])

main()