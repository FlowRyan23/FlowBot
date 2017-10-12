import numpy as np
import random

DATA_DIR = '../../FlowBot Training Data/'

blacklisted_input_tenors = [[0.0, 0.0, 92.78, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256.0, -3839.99, 16.66, -56, 16384, 0, -0.0, 0.0, 8.13, 0.0, -0.0, -0.0, 33, -254.53, 3840.09, 16.7, -60, -16380, 0, 0.01, 0.05, 7.78, -0.0, -0.0, 0.0, 33]]

blacklisted_output_tensors = [[128, 128, 128, 128, 0, 0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]


def process(file_name):
	reformat(file_name)
	features, n_read = read_all_ff(DATA_DIR + '/temp/' + file_name + '_formatted.txt')
	features, n_culled = cull_features(features)
	write_all(features, file_name)

	return n_read, n_culled


def reformat(file_name):
	src_path = DATA_DIR + 'raw/' + file_name + '.txt'
	dest_path = DATA_DIR + 'temp/' + file_name + '_formatted.txt'
	with open(src_path, 'r') as src_file:
		# read initial feature
		x = src_file.readline()
		y = src_file.readline()
		feature_separator = src_file.readline()
		feature_count = 0
		with open(dest_path, 'w') as ref_file:
			while feature_separator == '\n':
				feature_count += 1

				# take out unwanted chars
				x = x.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')
				y = y.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')

				# concat x and y with separator
				feature_string = x + '::::' + y + '\n'
				ref_file.write(feature_string)

				# read in next feature
				x = src_file.readline()
				y = src_file.readline()
				feature_separator = src_file.readline()

	return feature_count


def cull_features(features):
	spared_features = []
	n_culled = 0
	for feature in features:

		''' proved to be unsuccessful so far
		#cull black listed input tensors
		for bl_in_tensor in blacklisted_input_tenors:
			#iterating over every entry in input tensor
			for index in range(len(bl_in_tensor)):
				#if value differs from black listed tensor the feature is spared
				if feature[0][index] != bl_in_tensor[index]:
					break
				#feature is culled and count incremented
				feature = 0
				n_culled += 1
			#culled features do not need to be compared to other black listed tensors
			if feature == 0:
				break

		#cull black listed output tensors
		if feature != 0:
			for bl_out_tensor in blacklisted_output_tensors:
				for index in range(len(bl_in_tensor)):
					if feature[1][index] != bl_in_tensor[index]:
						break
					feature = 0
					n_culled += 1
				if feature == 0:
					break
		'''

		# cull all tensors with no ball info (all ball info =0 except height)
		ball_info_sum = 0
		for index in range(15):
			if index != 2:
				ball_info_sum += feature[0][index]
		if ball_info_sum == 0:
			n_culled += 1
			feature = 0

		# add non-culled features to res
		if feature != 0:
			spared_features.append(feature)

	return spared_features, n_culled


# requires the file at 'path' to in post processing format
def read_all_ff(path):
	features = []
	with open(path, 'r') as data_file:
		# read initial feature
		feature = data_file.readline()
		feature_count = 0
		while feature != '':
			feature_count += 1
			# split the feature string in x and y component
			x_y_tuple = feature.replace('\n', '').split('::::')
			x = convert_to_float_list(x_y_tuple[0])
			y = convert_to_float_list(x_y_tuple[1])
			if len(x) == 41 and len(y) == 14:
				features.append([x, y])
			else:
				print('read invalid feature at: ', feature_count)

			feature = data_file.readline()
	return features, feature_count


def read_all_rec(path):
	features = []
	with open(path, 'r') as data_file:
		# read initial feature
		feature = data_file.readline()
		feature_count = 0
		while feature != '':
			feature_count += 1
			# split the feature string in x and y component
			x_y_tuple = feature.replace('\n', '').split('::::')
			x = convert_to_float_list(x_y_tuple[0])
			y = convert_to_float_list(x_y_tuple[1])
			if len(x) == 41 and len(y) == 14:
				feature_x = []
				for val in x:
					feature_x.append([val])

				features.append([feature_x, y])
			else:
				print('read invalid feature at: ', feature_count)

			feature = data_file.readline()
	return features, feature_count


def write_all(features, file_name):
	dest_path = DATA_DIR + '/processed/' + file_name + '_processed.txt'
	with open(dest_path, 'a') as file:
		for feature in features:
			as_string = convert_to_feature_string(feature)
			file.write(as_string)


def convert_to_float_list(string):
	string_list = string.split()
	float_list = []
	for s in string_list:
		try:
			float_list.append(float(s))
		except ValueError:
			continue
	return float_list


def convert_to_feature_string(feature):
	x = str(feature[0])
	y = str(feature[1])

	# take out unwanted chars
	x = x.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')
	y = y.replace('[', '').replace(']', '').replace('\'', '').replace(',', "").replace('\n', '').replace('-0.0', '0').replace('0.0', '0')

	feature_string = x + '::::' + y + '\n'
	return feature_string


def get_feature_sets_ff(file_name, test_size=0.1):
	path = DATA_DIR + 'processed/' + file_name + '.txt'
	features, n = read_all_ff(path)
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(n * test_size)

	# notation: <start_index>:<end_index> (':,0' returns all elements of dim0)
	train_x = list(features[:, 0][:-testing_size])
	train_y = list(features[:, 1][:-testing_size])

	test_x = list(features[:, 0][-testing_size:])
	test_y = list(features[:, 1][-testing_size:])

	return train_x, train_y, test_x, test_y


# todo copy get_feature_sets_ff except train_x and test_x need to have every value inside the individual features put into a list of length 1
def get_feature_sets_recurrent(file_name, test_size=0.1):
	path = DATA_DIR + 'processed/' + file_name + '.txt'
	features, n = read_all_rec(path)

	features = np.array(features)
	testing_size = int(n * test_size)

	# notation: <start_index>:<end_index> (':,0' returns all elements of dim0)
	train_x = list(features[:, 0][:-testing_size])
	train_y = list(features[:, 1][:-testing_size])

	test_x = list(features[:, 0][-testing_size:])
	test_y = list(features[:, 1][-testing_size:])

	return train_x, train_y, test_x, test_y


if __name__ == '__main__':
	num_read, num_culled = process('training_data_1507408321')
	print('Read: ', num_read)
	print('Culled: ', num_culled)
	print('Retained: ', num_read - num_culled)
