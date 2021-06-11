#!/usr/bin/env python3

"""Utils file to support scripts with useful functions

	:Date: 05.2021
	:Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

""" IMPORTS """
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
from types import DynamicClassAttribute
import numpy as np
from typing import Tuple, List, Dict
from nptyping import NDArray
from keras.preprocessing.image import load_img, img_to_array


""" FUNCTIONS """
def optimize_batch_size(
    path_to_images: str
) -> int:
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


def pop_batch(
    batch: List[str], 
    full_list: List[str]
) -> List[str]:
	"""Remove chosen images batch from list of all images 

	:param batch: Chosen (sampled) batch from full_list
	:type batch: list
	:param full_list: List of images from which the batch is chosen
	:type full_list: list

	:return: List with removed batch
	:rtype: list
	"""
	return list(filter(lambda i: i not in batch, full_list))


def list_shuffle_fission(
    full_list: List[str], 
    k: int = 200
) -> Tuple[List[str], List[str]]:
	"""Choose batch and remove it from full_list, then return both

	:param full_list: List to choose from
	:type full_list: list
	:param k: Population of a batch, defaults to 200
	:type k: int

	:return: Batch of k images && List with removed batch
	:rtype: Tuple[list, list]
	"""
	lvl_batch = random.sample(full_list, k=k)
	residual_list = pop_batch(lvl_batch, full_list)
	return lvl_batch, residual_list


def prebatch_files(
    files: List[List[str]], 
    max_batch_population: int
) -> Tuple[List[List[str]], List[str]]: # 2-dim
	"""Make batches from images with `max_batch_population` each

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


def pop_testing_set(all_files: List[List[str]], batch_k: int = 1064) -> Tuple[List[List[str]], List[List[str]]]:
	"""Pop batch for testing from training data

	:param all_files: All training files groupped by category
	:type all_files: List[List[str]]
	:param batch_k: batch_k of testing set, defaults to 1064
	:type batch_k: int, optional
	:return: Return 1st=Testing and 2nd=Training set
	:rtype: Tuple[List[List[str]], List[List[str]]]
	"""
	test_files = []
	train_files = []
	
	for lvl in all_files:
		test_lvl, train_lvl = list_shuffle_fission(lvl, batch_k)
		test_files.append(test_lvl)
		train_files.append(train_lvl)
  
	return ( test_files, train_files )


def check_compression_level(
    filename: str
) -> int:
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


def prepare_fragmented_images(
    base_path: str,
    files: List[str],
    classes: Dict[str, int],
    im_height: int,
    num_channels: int,
    nfrags: int
) -> Tuple[NDArray, NDArray]:
	"""Make batches from list of file_names, divide into fragments and return List of floats and labels

	:param base_path: Absolute path to directory where images are
	:type base_path: str
	:param files: List of files to preprocess
	:type files: List[str]
	:param classes: Dictionary of classes to exchange compression level into numerical label, values 0..<num_of_classes>
	:type classes: dict
	:param im_height: Image height, defaults to 224
	:type im_height: int, optional
	:param num_channels: Number of color channels, defaults to 3
	:type num_channels: int, optional
	:param nfrags: Number of fragments, defaults to 8
	:type nfrags: int, optional

	:return: Dataset as normalized floats && labels to each image
	:rtype: Tuple[NDArray, NDArray]
	"""
	step = int(im_height / nfrags)
	dataset = np.ndarray(
		shape=(int(len(files)*(nfrags**2)), step, step, num_channels),
		dtype=np.float32
	)
	labels = np.ndarray(
		shape=(int(len(files)*(nfrags**2))),
		dtype=np.int32
	)

	for idx, _f in enumerate(files):
		label = check_compression_level(_f)
		path = os.path.join(base_path, str(label), _f)

		image = load_img(path)
		image = img_to_array(image, data_format='channels_last')

		frag_num = 0
		for n in range(nfrags):
			for k in range(nfrags):

				frag = image[ n*step : (n+1)*step, k*step : (k+1)*step ]

				# loading data to arrays
				dataset[ (nfrags**2)*idx + frag_num ] = frag
				
				labels[ (nfrags**2)*idx + frag_num ] = classes[str(label)]
				frag_num += 1   

	# normalize dataset to range 0.0..1.0
	dataset = dataset / 255.0

	return dataset, labels


def prepare_full_images(
    base_path: str,
    files: List[str],
    classes: Dict[str, int],
    im_height: int,
    im_width: int,
    num_channels: int,
) -> Tuple[NDArray, NDArray]:
	"""Make batches from list of file_names and return List of floats and labels

	:param base_path: Absolute path to directory where images are
	:type base_path: str
	:param files: List of files to preprocess
	:type files: List[str]
	:param classes: Dictionary of classes to exchange compression level into numerical label, values 0..<num_of_classes>
	:type classes: dict
	:param fragment: To fragment images [ 0 ] or not [ 1 ]
	:type fragment: int
	:param im_height: Image height, defaults to 224
	:type im_height: int, optional
	:param im_width: Image width, defaults to 224
	:type im_width: int, optional
	:param num_channels: Number of color channels, defaults to 3
	:type num_channels: int, optional

	:return: Dataset as normalized floats && labels to each image
	:rtype: Tuple[NDArray, NDArray]
	"""
	dataset = np.ndarray(
		shape=(int(len(files)), im_height, im_width, num_channels),
		dtype=np.float32
	)
	labels = np.ndarray(
		shape=(int(len(files))),
		dtype=np.int32
	)
 
	for idx, _f in enumerate(files):
		label = check_compression_level(_f)
		path = os.path.join(base_path, str(label), _f)

		image = load_img(path)
		image = img_to_array(image, data_format='channels_last')

		dataset[ idx ] = image
		labels[ idx ] = classes[str(label)]

	# normalize dataset to range 0.0..1.0
	dataset = dataset / 255.0

	return dataset, labels


def prepare_images_with_labels(
    base_path: str,
    files: List[str],
    classes: Dict[str, int],
    fragment: int,
    im_height: int = 224,
    im_width: int = 224,
    num_channels: int = 3,
    nfrags: int = 8
) -> Tuple[NDArray, NDArray]:
	"""Wrapper: Make batches from list of file_names and return List of floats and labels

	:param base_path: Absolute path to directory where images are
	:type base_path: str
	:param files: List of files to preprocess
	:type files: List[str]
	:param classes: Dictionary of classes to exchange compression level into numerical label, values 0..<num_of_classes>
	:type classes: dict
	:param fragment: To fragment images [ 0 ] or not [ 1 ]
	:type fragment: int
	:param im_height: Image height, defaults to 224
	:type im_height: int, optional
	:param im_width: Image width, defaults to 224
	:type im_width: int, optional
	:param num_channels: Number of color channels, defaults to 3
	:type num_channels: int, optional
	:param nfrags: Number of fragments, defaults to 8
	:type nfrags: int, optional

	:return: Dataset as normalized floats && labels to each image
	:rtype: Tuple[NDArray, NDArray]
	"""
	if fragment:
		return prepare_fragmented_images(base_path, files, classes, im_height, num_channels, nfrags)
	else:
		return prepare_full_images(base_path, files, classes, im_height, im_width, num_channels)


def calculate_batch_and_save(
    base_path: str,
    batch: List[str], 
    save_name: str, 
    classes: Dict[str, int],
    fragment: int = 0
):
	"""Take batch, make it Float matrix with labels and save it to .npz file

	:param batch: List of filenames
	:type batch: list
	:param save_name: Name to save the batch
	:type save_name: str
	:param classes: List of classes
	:type classes: list
	:param fragment: Whether to fragment data [ 1 ] or not [ 0 ], defautls to 0
 	:type fragment: int
	"""
 	# read all files
	( dataset, labels ) = prepare_images_with_labels(base_path, batch, classes, fragment)

	# save the database and labels to *.npz files
	np.savez(save_name, dataset=dataset, labels=labels)
 
	print(f"Saved to file: {os.path.basename(save_name)} => {dataset.shape=}, {labels.shape=}")	


def calculate_mixed_batch_and_save(
	path_ds1: str,
	path_ds2: str,
 	bds1: List[str],
	bds2: List[str],
	save_name: str,
	classes: Dict[str, int],
	fragment: int = 0
) -> None:
	"""Calculate two batches and save as train and test set

	:param path_train: Path to train set folder
	:type path_train: str
	:param path_test: Path to test set folde
	:type path_test: str
	:param btrain: Train batch
	:type btrain: List[str]
	:param btest: Test batch
	:type btest: List[str]
	:param save_name: Name to save the batch
	:type save_name: str
	:param classes: Classes dictionary
	:type classes: Dict[str, int]
	:param fragment: Whether to fragment data [ 1 ] or not [ 0 ], defautls to 0
	:type fragment: int, optional
 	"""
	# read all files
	( nsdtest, nsltest ) = prepare_images_with_labels(path_ds1, bds1, classes, fragment)
	( ntdtest, ntltest ) = prepare_images_with_labels(path_ds2, bds2, classes, fragment)

	# save the database and labels to *.npz files
	np.savez(
     	save_name, 
     	nsdtest=nsdtest,
      	nsltest=nsltest,
       	ntdtest=ntdtest, 
      	ntltest=ntltest
    )
	print(f"Saved to file: {os.path.basename(save_name)} => {nsdtest.shape=}, {nsltest.shape=}, {ntdtest.shape=}, {ntltest.shape=}")


def banner(
    text: str, *, 
    length: int = 65, 
    frame_char: str = '#'
) -> str:
	"""Print banner in terminal

	:param text: Text to display in banner
	:type text: str
	:param length: Width of the banner in columns, defaults to 65
	:type length: int, optional
	:param frame_char: Character to make frame of, defaults to '#'
	:type frame_char: str, optional
	:return: Readymade banner to print out on the screen 
	:rtype: str
	"""
	stext = ' %s ' % text
	mbanner = frame_char*2 + stext.center(length-4) + frame_char*2
	fframe = frame_char*length
	eframe = frame_char*2 + ' '*(length-4) + frame_char*2
	banner = f"{fframe}\n{eframe}\n{mbanner}\n{eframe}\n{fframe}\n\n"
	return banner


def continue_or_quit():
    """Wait for user input to continue or quit if requested"""
    q = input("Press q to quit, Enter to continue.. ")
    if q in ['q', 'quit', 'exit']:
        sys.exit(0)
    print("\n")

    
def check_prediction(
    predicted_array: List[float],
    real: int
) -> Tuple[bool, float]:
	"""Check if prediction was successful
 
	:param predicted_array: Predicted array of probabilities 
	:type predicted_array: List[float]
	:param real: Real label of image
	:type real: int
 	"""
	predicted = np.argmax(predicted_array)
	if predicted == real:
		return True, predicted_array[predicted]
	else:
		return False, predicted_array[predicted]


""" MAIN """
if __name__ == "__main__":
    print(banner("Hello"))
