#!/usr/bin/env python3

"""Load undivided set of images, 
	split into categories by name and put it in category folders

	:Date: 04.2021
	:Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os, shutil
from utils import check_compression_level

base_path = os.getcwd()
folder_name = '224x224_NASA_classed'
base_folder = '224x224_NASA_dataset'

def main():
	os.chdir(base_path)

	if not os.path.exists(os.path.join(base_path, folder_name)):
		os.mkdir(os.path.join(base_path, folder_name))

	files = os.listdir(base_folder)

	try:
		for file in files:
			label = check_compression_level(file)
			
			if not os.path.exists(os.path.join(base_path, folder_name, str(label))):
				os.mkdir(os.path.join(base_path, folder_name, str(label)))

			shutil.move(os.path.join(base_path, base_folder, file), os.path.join(base_path, folder_name, str(label)))
	except Exception as e:
		print(e)


""" MAIN """
if __name__ == "__main__":
	main()