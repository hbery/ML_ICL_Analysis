import os, shutil

base_path = os.getcwd()
folder_name = '224x224_NASA_classed'
base_folder = '224x224_NASA_dataset'

def check_compression_level(filename: str) -> int:
	basename, ext = filename.split('.')
	tmp = basename.split('_')
	if len(tmp) == 1:
		return 100
	else:
		return int(tmp[1])

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


if __name__ == "__main__":
	main()