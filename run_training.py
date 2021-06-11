#!/usr/bin/env python3

"""Training runnner for model

    :Date: 05.2021
    :Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json, time
import numpy as np
from itertools import zip_longest
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Softmax

from utils import banner
from model import build_model


def main():
    
    """ ~~~~ PREPARE DATA ~~~~ """
    if len(sys.argv) < 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <folder with batches> <'model_name'>")
        sys.exit(-1)
    
    folder = os.path.basename(sys.argv[1])
    base_path = os.path.abspath(folder)
    model_name = os.path.basename(sys.argv[2])
    model_path = os.path.join(os.getcwd(), "models", sys.argv[2])
    default_line_length = 65
    tmp_save = 1
    
    dir_files = os.listdir(base_path)
    class_file = list(filter(lambda file: "classes" in file, dir_files))[0]
    test_files = list(filter(lambda file: "test" in file, dir_files))
    train_files = list(filter(lambda file: "train" in file, dir_files))

    with open(os.path.join(base_path, class_file)) as fjson:
        classes = json.load(fjson)
    
    test_files.sort()
    train_files.sort()
    
    print(banner("INITIAL DATA", length=default_line_length))
    print(f"Model path:\n\t{model_path}\n")
    print(f'{"TRAIN_FILES"}\t{"TEST_FILES"}'.center(default_line_length))
    tlen = len(test_files[0])
    for trainf, testf in zip_longest(train_files, test_files):
        print(f'{trainf}\t{testf or " "*tlen}'.center(default_line_length))
    
    print("CLASSES:")
    print(f"{classes}".center(default_line_length))
    print(f'Checkpoint saves: {"True" if tmp_save else "False"}\n')    
    
    
    print(banner("MODEL"))
    
    model = build_model(len(classes.keys()), name="Model_BIQA_NASA")
    print(model.summary(line_length=default_line_length))
    print()
    
    q = input("Press q to quit, Enter to continue.. ")
    if q in ['q', 'quit', 'exit']:
        sys.exit(0)
    print("\n")
    
    """ ~~~~ BATCH LOOP ~~~~ """
    score = []
    history = []
    
    print(banner("TRAINING", length=default_line_length))
    
    start = time.time()
    epoch_num = 6
    for bnum, batch in enumerate(train_files):
        batch_time = time.time()
        with np.load(os.path.join(base_path, batch)) as data_batch:
    # Loading from *.npz
            x = data_batch['dataset']
            y = data_batch['labels']
        print(f"┏━ FILE: {batch} :\n┃ train_set:\t{x.shape} » train_labels:\t{y.shape}")
        print("┃ "+"_"*63)
    # Splitting for training and evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        n = 0
        for train_indices, test_indices in skf.split(x, y):    
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            n += 1
    # Training..
            training = time.time()
            fit_history = model.fit(x=x_train, y=y_train, batch_size=96, epochs=epoch_num, verbose=0)
            print(f"┃ ⤷Fold_{n} training ({epoch_num} epochs): {round(time.time() - training)} seconds..")
            fit_loss, fit_accuracy = fit_history.history["loss"][-1], fit_history.history["accuracy"][-1]
            print(f"┃ \t↻Fold_{n} training: {fit_loss=:.2f}, {fit_accuracy=:.2f}")
            history.append(fit_history.history)
    # Evaluating..
            eval_loss, eval_accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(f"┃ \t↺Fold_{n} evaluation: {eval_loss=:.2f}, {eval_accuracy=:.2f}")
            score.append({
                "fit_loss": fit_loss,
                "fit_accuracy": fit_accuracy,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy
            })
    # Ending..
        print(f'┗━ FILE: {batch} time: {time.strftime("%H h %M min %S sec", time.gmtime(round(time.time() - batch_time)))}')
        if tmp_save:
            checkpoint = f"{model_path}{bnum:02}"
            model.save(checkpoint)
            print(f"🗘 Model's checkpoint saved as:\n\t{checkpoint}".center(default_line_length))
        print("="*default_line_length)
    # Summary time spent training and evaluating
    print(f'⇶ Full training took: {time.strftime("%H h %M min %S sec", time.gmtime(round(time.time() - start)))}')
    
    """ ~~~~ SAVE MODEL ~~~~ """
    model.save(model_path)
    print(f"⮔ Model saved as:\n\t{model_path}\n".center(default_line_length))


    tstats_path = os.path.join(base_path, f'{model_name}_train_stats.json')
    thistory_path = os.path.join(base_path, f'{model_name}_train_history.json')
    # Save score and history for plotting
    with open(tstats_path, 'w') as fjson:
        json.dump(score, fjson)
    
    with open(thistory_path, 'w') as fjson:
        json.dump(history, fjson)

    print(f"⮔ Train data saved as:\n\t{tstats_path}\n\t{thistory_path}\n".center(default_line_length))

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
            nasa_predictions.extend(prob_model.predict(test_batch["nsdtest"]))
            nature_predictions.extend(prob_model.predict(test_batch["ntdtest"]))
          
    # Save data for plotting
    stats_path = os.path.join(base_path, f'{model_name}_stats.npz')
    np.savez(stats_path,
        nasa_predictions=nasa_predictions,
        nasa_labels=nasa_labels,
        nature_predictions=nature_predictions,
        nature_labels=nature_labels
    )
    print(f"⮔ Statistics saved as:\n\t{stats_path}".center(default_line_length))


""" MAIN """
if __name__ == "__main__":
    main()