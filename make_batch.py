# imports
import os
import random
import json
import numpy as np
from typing import Tuple, List
from keras.preprocessing.image import load_img, img_to_array

# "GLOBALS"
IM_H = 224
IM_W = 224
CHAN = 3
NFRAGS = 8
STEP = int(IM_H / NFRAGS)

def optimize_batch_size(path_to_images: str) -> int:
	"""Count all files and declare "optimal" [very narrow case] maximum batch size

	:param path_to_images: Path to the categorized images directory: path_to_images/class_dir/[images..]
	:type path_to_images: str
 
	:return: Maximum batch size
	:rtype: int
	"""
	classes = os.listdir(path_to_images)
	num_classes = len(classes)
	num_images_per_class = len(os.listdir(os.path.join(path_to_images, classes[0]))) # assume equal class population
	num_batches = (num_classes * num_images_per_class) // 1500
	
	return int(np.ceil( num_images_per_class / (num_batches + 1)))

def pop_batch(batch: List[str], full_list: List[str]) -> List[str]:
	"""Remove chosen images batch from list of all images 

	:param batch: Chosen (sampled) batch from full_list
	:type batch: list
 
	:param full_list: List of images from which the batch is chosen
	:type full_list: list

	:return: List with removed batch
	:rtype: list
	"""
	return list(filter(lambda i: i not in batch, full_list))

def list_shuffle_fission(full_list: List[str], k: int = 200) -> Tuple[List[str], List[str]]:
	"""Choose batch and remove it from full_list, then return both

	:param full_list: List to choose from
	:type full_list: list

	:param k: Population of a batch, defaults to 200
	:type k: int

	:return: Batch of k images && List with removed batch
	:rtype: Tuple[list, list]
	"""
	lvl_batch = random.sample(full_list, k=k)
	full_list = pop_batch(lvl_batch, full_list)
	return lvl_batch, full_list


def prebatch_files(files: List[List[str]], max_batch_population: int) -> Tuple[List[List[str]], List[str]]: # 2-dim
	"""Make batches from images with :max_batch_population: each

	:param files: List of all files
	:type files: List[str]
	
 	:param max_batch_population: Maximum batch population
	:type max_batch_population: int
	
 	:return: List of batches and last non-full batch
	:rtype: Tuple[List[List[str]], List[str]]
	""" 
	nbatches = (len(files[0]) // max_batch_population)
	batches = []
	res_batch = []

	for _ in range(nbatches):
		batch = []
		fcopy = []
		
		for lvl in files:
			lvl_batch, tmplist = list_shuffle_fission(lvl, max_batch_population)
			batch.extend(lvl_batch)
			fcopy.append(tmplist)

		batches.append(batch)
		files = fcopy

	[res_batch.extend(x) for x in files]
	random.shuffle(res_batch)
	
	return batches, res_batch 


def check_compression_level(filename: str) -> int:
	"""Check compression level

	:param filename: Filename in format: <filename>_<compression_level>.<extension>
	:type filename: str
 
	:return: Compression level
	:rtype: int
	"""
	basename, _ = filename.split('.')
	tmp = basename.split('_')
	if len(tmp) == 1:
		return 100
	else:
		return int(tmp[1])


def prepare_images_with_labels(base_path: str, files: List[str], classes: dict) -> Tuple[List[List[List[float]]], List[int]]:
	"""Make batches from list of file_names and return List of floats and labels

	:param base_path: Absolute path to directory where images are
	:type base_path: str
 
	:param files: List of files to preprocess
	:type files: List[str]
 
	:param classes: Dictionary of classes to exchange compression level into numerical label, values 0..<num_of_classes>
	:type classes: dict
 
	:return: Dataset as normalized floats && labels to each image
	:rtype: Tuple[List[List[List[float]]], List[int]]
	"""
	dataset = np.ndarray(
		shape=(int(len(files)*(NFRAGS**2)), STEP, STEP, CHAN),
		dtype=np.float32
	)
	y = np.ndarray(
		shape=(int(len(files)*(NFRAGS**2))),
		dtype=np.int32
	)

	for idx, _f in enumerate(files):
		label = check_compression_level(_f)
		path = os.path.join(base_path, str(label), _f)

		image = load_img(path)
		image = img_to_array(image, data_format='channels_last')

		frag_num = 0
		for n in range(NFRAGS):
			for k in range(NFRAGS):

				frag = image[ n*STEP : (n+1)*STEP, k*STEP : (k+1)*STEP ]

				# loading data to arrays
				dataset[ (NFRAGS**2)*idx + frag_num ] = frag
				y[ (NFRAGS**2)*idx + frag_num ] = classes[str(label)]
				frag_num += 1

	# normalize dataset to range 0.0..1.0
	dataset = dataset / 255.0

	return dataset, y

def calculate_batch_and_save(batch: List[str], save_name: str, classes: dict):
	"""Take batch, make it Float matrix with labels and save it to .npy file

	:param batch: List of filenames
	:type batch: list
 
	:param save_name: Name to save the batch
	:type save_name: str
 
	:param classes: List of classes
	:type classes: list
	"""
 	# read all files
	( x, y ) = prepare_images_with_labels(base_path, batch, classes)

	# save the database and labels to *.npz files
	np.savez(save_name, dataset=x, labels=y)


if __name__ == "__main__":

	num_classes = 15
	# variables
	base_path = os.path.abspath(f'./224x224_{num_classes}_compression_levels')
	# base_path = os.path.abspath('./224x224_15_compression_levels')
	all_files = []
	out_files = os.path.abspath(f'./out_files_{num_classes}cat')


	# Make batches into separate files
	class_list = os.listdir(base_path)
	[ all_files.append(os.listdir(os.path.join(base_path, directory))) for directory in os.listdir(base_path) ]

	# make classes into numbers from 0-num(classes)
	classes = {}
	for idx, label in enumerate(sorted(list(set(class_list)), key=lambda x: int(x))):
		classes[label] = idx

	with open(os.path.join(out_files, f"classes_{num_classes}cat.json"), "w") as fclass:
		json.dump(classes, fclass)

	( batches, res_batch ) = prebatch_files(all_files, optimize_batch_size(base_path))

	cnt = 1
	for batch in batches:
		calculate_batch_and_save(batch, os.path.join(out_files, f'dataset_labels_{num_classes}cat_{cnt:02}.npz'), classes)
		cnt += 1

	calculate_batch_and_save(res_batch, os.path.join(out_files, f'dataset_labels_{num_classes}cat_{cnt:02}.npz'), classes)
