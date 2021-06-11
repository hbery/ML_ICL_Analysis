#!/usr/bin/env python3

"""Script for testing model

    :Date: 06.2021 
    :Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Softmax

from utils import banner


def main():
	
	""" ~~~~ PREPARE DATA ~~~~ """
	if len(sys.argv) < 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <folder with batches> <'model_name'>")
        sys.exit(-1)

    folder = os.path.basename(sys.argv[1])
    base_path = os.path.abspath(folder)
    model_name = os.path.basename(sys.argv[2])
    model_path = os.path.join(os.getcwd(), "models", model_name)
    default_line_length = 65
    
    dir_files = os.listdir(base_path)
    test_files = list(filter(lambda file: "test" in file, dir_files))

    print(banner("MODEL"))
    model = load_model(model_path)
    print(model.summary(line_length=default_line_length))
    print()


	""" ~~~~ TEST MODEL'S ACCURACY ~~~~ """
    print(banner("TESTING", length=default_line_length))

    print("⇊ Adding Softmax Layer to model")
    prob_model = Sequential([model, Softmax()])

    nasa_predictions = []
    nasa_labels = []
    nature_predictions = []
    nature_labels = []

    for test_file in test_files:
    # Loading from *.npz
        with np.load(os.path.join(base_path, test_file)) as test_batch:
    # Storing real labels
            nasa_labels.extend(test_batch["nsltest"])
            nature_labels.extend(test_batch["ntltest"])
    # Predicting labels and storing
            nasa_predictions.extend(model.predict(test_batch["nsdtest"]))
            nature_predictions.extend(model.predict(test_batch["ntdtest"]))
          
    # Save data for plotting
    stats_path = os.path.join(base_path, f'{model_name}_stats.npz')
    np.savez(stats_path,
        nasa_predictions=nasa_predictions,
        nasa_labels=nasa_labels,
        nature_predictions=nature_predictions,
        nature_labels=nature_labels
    )
    print(f"⮔ Statistics saved as: {stats_path}".center(default_line_length))


"""MAIN """
if __name__ == "__main__":
	main()
