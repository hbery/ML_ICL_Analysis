#!/usr/bin/env python3

"""Make .npz batches out of images to ease the training

	:Date: 04.2021
	:Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os, json
from utils import (
    prebatch_files,
    calculate_batch_and_save
)


def make_batch():
    
	num_classes = 6
 
	# variables
	base_path = os.path.abspath(f'./224x224_{num_classes}_compression_levels')
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

	( batches, res_batch ) = prebatch_files(all_files, 200)

	cnt = 1
	for batch in batches:
		calculate_batch_and_save(base_path, batch, os.path.join(out_files, f'dataset_labels_full_{num_classes}cat_{cnt:02}.npz'), classes)
		cnt += 1

	calculate_batch_and_save(base_path, res_batch, os.path.join(out_files, f'dataset_labels_full_{num_classes}cat_{cnt:02}.npz'), classes)


""" MAIN """
if __name__ == "__main__":
    make_batch()
