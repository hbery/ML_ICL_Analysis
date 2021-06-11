#!/usr/bin/env python3

"""Script to plot charts for model training and testing statistics

    :Date: 06.2021
    :Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os, sys
import json
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import continue_or_quit, check_prediction


def main():

	# Constant values
	div_batches = 1
	b_ttstats = 1
	b_trstats = 1
	b_trhistory = 1
	num_train_batches = 8
 
	figW = 12
	figH = 9
 
	### ~~~~ PREPARE DATA ~~~~ ###
	if len(sys.argv) < 2:
		print(f"Usage: {os.path.basename(sys.argv[0])} <'model_name'>")
		sys.exit(-1)

	base_path = os.path.abspath("./statistics")
	model_name = os.path.basename(sys.argv[1])
	class_file = os.path.join(base_path, "classes_6cat.json")

	with open(class_file, 'r') as cfile:
		classes = json.load(cfile)

	num_classes = len(classes.keys())

	"""list containing dicts with arrays(len=num_epochs) with certain stats:
		"loss" - loss of training per epoch
		"accuracy" - accuracy of training per epoch
	"""
	training_history = f"{model_name}_train_history.json"

	"""list containing dicts with fields:
		"fit_loss" - last epoch in fold loss
		"fit_accuracy" - last epoch in fold accuracy
		"eval_loss" - fold eval loss
		"eval_accuracy" - fold eval accuracy
	"""
	training_stats = f"{model_name}_train_stats.json"

	"""file containing np.arrays: [very specific for our case]
		"nasa_predictions" - labels predicted by trained model | NASA dataset
		"nasa_labels" - real labels | NASA dataset
		"nature_predictions" - labels predicted by trained model | NATURE dataset
		"nature_labels" - real labels | NATURE dataset
	"""
	testing_stats = f"{model_name}_stats.npz"

	### ~~~~ LOAD DATA ~~~~ ###


	xtrainlabels = [ f"b{i+1:02}" for i in range(num_train_batches) ]


	# Load trainig history
	if b_trhistory:
		print("Preparing training history")

		with open(os.path.join(base_path, training_history), 'r') as thjson:
			hst_dicts = json.load(thjson)

		offset = 0
		step = len(hst_dicts[0]['loss']) * 5
		loss_epochs = []
		accuracy_epochs = []
		for hst in hst_dicts:
			loss_epochs.extend( hst['loss'] )
			accuracy_epochs.extend( hst['accuracy'] )

		# Plot history
		thax = pd.DataFrame.from_dict(
			{"loss": loss_epochs, "accuracy": accuracy_epochs}
			).plot(
				figsize=[figW, figH],
				title="Training metrics history",
				xlabel="Batches",
				secondary_y=( "loss", ),
				mark_right=True
			)
		thax.set_ylim([-1.1, 1.1])
		thax.right_ax.set_ylim([-0.1, 4.1])

		if div_batches:
			[ thax.axvline(x=ix, color='y', linestyle='--') if (ix % step) == 0 else None for ix in range(len(loss_epochs)) ]
			offset = step/2

		thax.set_xticks(
			np.arange(
				start=offset,
				stop=len(loss_epochs),
				step=step)
		)

		thax.set_xticklabels( xtrainlabels )
		thax.legend(loc='center left')
		thax.right_ax.legend(loc='center right')

		# Wait for user
		# continue_or_quit()

	# Load training statistics
	if b_trstats:
		print("Preparing training statistics")
        
		with open(os.path.join(base_path, training_stats), 'r') as tsjson:
			stats_dicts = json.load(tsjson)

		offset = 0
		step = len(stats_dicts) / num_train_batches

		# Plot training statistics
		tsax = pd.DataFrame.from_dict(
			stats_dicts
			).plot(
				figsize=[figW, figH],
				title="Training and Testing statistics",
				xlabel="Batches",
				secondary_y=( "fit_loss", "eval_loss" ),
				mark_right=True
			)
		tsax.set_ylim([-1.0, 1.1])
		tsax.right_ax.set_ylim([-0.1, 4.1])

		if div_batches:
			[ tsax.axvline(x=ix, color='y', linestyle='--') if (ix % step) == 0 else None for ix in range(len(stats_dicts)) ]
			offset = step/2

		tsax.set_xticks(
			np.arange(
				start=offset,
				stop=len(stats_dicts),
				step=step)
		)
		tsax.set_xticklabels( xtrainlabels )
		tsax.legend(loc='center left')
		tsax.right_ax.legend(loc='center right')

		# Wait for user
		# continue_or_quit()

	# Load testing statistics
	if b_ttstats:
		print("Preparing testing statistics")
		bar_width = 0.2
		
		with np.load(os.path.join(base_path, testing_stats), 'r') as ttstats:
			nasa_predictions    = ttstats["nasa_predictions"]
			nasa_labels         = ttstats["nasa_labels"]
			nature_predictions  = ttstats["nature_predictions"]
			nature_labels       = ttstats["nature_labels"]

		# Prepare figure
		fig, axes = plt.subplots(
      						nrows=1,
							ncols=2,
							sharey=True,
       						sharex=True,
							squeeze=False,
							figsize=[figW*1.8, figH],
							tight_layout=True,
							num="Model prediction statistics"
       	)

		(nsax, ntax) = axes.flatten()

		fig.suptitle("Compression level prediction statistics")
		fig.supxlabel("JPEG compression level")
		fig.supylabel("Percent of images in set [ % ]")
		
		# NASA
		nspos_count = [0]*num_classes
		nsneg_count = [0]*num_classes
		nspos_pr = [0.0]*num_classes
		nsneg_pr = [0.0]*num_classes
		nslen = len(nasa_labels)
		
		for idx in range(nslen):
			pair = check_prediction(nasa_predictions[idx], nasa_labels[idx])
			if pair[0]:
				nspos_count[nasa_labels[idx]] += 1
				nspos_pr[nasa_labels[idx]] += pair[1]
			else:
				nsneg_count[nasa_labels[idx]] += 1
				nsneg_pr[nasa_labels[idx]] += pair[1]

		nspos_pr = [ f"{proc / nspos_count[i] * 100:.2f}%" if nspos_count[i] else "0.00%" for i, proc in enumerate(nspos_pr) ]
		nsneg_pr = [ f"{proc / nsneg_count[i] * 100:.2f}%" if nsneg_count[i] else "0.00%" for i, proc in enumerate(nsneg_pr) ]
		nspos_count = [ clx / (nslen / num_classes) * 100 for clx in nspos_count ]
		nsneg_count = [ clx / (nslen / num_classes) * 100 for clx in nsneg_count ]

		xnspos = [ x + 1 for x, _ in enumerate(nspos_count)]
		xnsneg = [ x + 1 + bar_width for x, _ in enumerate(nsneg_count)]

		
		nsax.set_title("NASA test dataset")

		tick_pos = [val + bar_width / 2 for val in xnspos]
		nsax.set_xticks(tick_pos)
		nsax.set_xticklabels(classes.keys())
		nsax.set_xlim([0, 7])

		nsposb = nsax.bar(xnspos, nspos_count, label="predicted", width=bar_width, color='g')
		nsnegb = nsax.bar(xnsneg, nsneg_count, label="not predicted", width=bar_width, color='r')

		nsax.bar_label(nsposb, nspos_pr, padding=2, rotation='vertical')
		nsax.bar_label(nsnegb, nsneg_pr, padding=2, rotation='vertical')

		nsax.legend(loc='upper right')

		# NATURE
		ntpos_count = [0]*num_classes
		ntneg_count = [0]*num_classes
		ntpos_pr = [0.0]*num_classes
		ntneg_pr = [0.0]*num_classes
		ntlen = len(nature_labels)
		
		for idx in range(ntlen):
			pair = check_prediction(nature_predictions[idx], nature_labels[idx])
			if pair[0]:
				ntpos_count[nature_labels[idx]] += 1
				ntpos_pr[nature_labels[idx]] += pair[1]
			else:
				ntneg_count[nature_labels[idx]] += 1
				ntneg_pr[nature_labels[idx]] += pair[1]

		ntpos_pr = [ f"{proc / ntpos_count[i] * 100:.2f}%" if ntpos_count[i] else "0.00%" for i, proc in enumerate(ntpos_pr) ]
		ntneg_pr = [ f"{proc / ntneg_count[i] * 100:.2f}%" if ntneg_count[i] else "0.00%" for i, proc in enumerate(ntneg_pr) ]
		ntpos_count = [ clx / (ntlen / num_classes) * 100 for clx in ntpos_count ]
		ntneg_count = [ clx / (ntlen / num_classes) * 100 for clx in ntneg_count ]

		xntpos = [ x + 1 for x, _ in enumerate(ntpos_count)]
		xntneg = [ x + 1 + bar_width for x, _ in enumerate(ntneg_count)]

		ntax.set_title("NATURE test dataset")

		tick_pos = [val + bar_width / 2 for val in xntpos]
		ntax.set_xticks(tick_pos)
		ntax.set_xticklabels(classes.keys())
		ntax.set_xlim([0, 7])

		ntposb = ntax.bar(xntpos, ntpos_count, label="predicted", width=bar_width, color='g')
		ntnegb = ntax.bar(xntneg, ntneg_count, label="not predicted", width=bar_width, color='r')

		ntax.bar_label(ntposb, ntpos_pr, padding=2, rotation='vertical')
		ntax.bar_label(ntnegb, ntneg_pr, padding=2, rotation='vertical')

		ntax.legend(loc='upper right')

	# Show all charts
	plt.show()


""" MAIN """
if __name__ == "__main__":
	main()