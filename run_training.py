import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from model import build_model


def main():
    
    """ ~~~~ PREPARE DATA ~~~~ """
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <folder with batches>")
        sys.exit(-1)
    folder = sys.argv[1]
    base_path = os.path.abspath(folder)
    cwd = os.getcwd()
    model_name = os.path.join(cwd, "models", folder[2:].split('_')[0])

    tmp_save = 1

    numpy_batches = os.listdir(base_path)
    class_file = [fl for fl in numpy_batches if "classes" in fl][0]
    numpy_batches.remove(class_file)
    with open(os.path.join(base_path, class_file)) as fjson:
        classes = json.load(fjson)
    numpy_batches.sort()
    
    model = build_model(len(classes.keys()), name="Model_BIQA_NASA-train_NATURE-test")
    print(model.summary())
    print()
    
    """ ~~~~ BATCH LOOP ~~~~ """
    print(f"Classes: {classes}")
    start = time.time()
    for batch in numpy_batches:
        batch_time = time.time()
        with np.load(os.path.join(base_path, batch)) as data_batch:
    # Loading from *.npz
            x = data_batch['dtrain']
            y = data_batch['ltrain']
            xt = data_batch['dtest']
            yt = data_batch['ltest']
        print(f"┏━ FILE: {batch} =>\n┃ train_set:\t{x.shape} » train_labels:\t{y.shape}\n┃ test_set:\t{xt.shape} » test_labels:\t{yt.shape}")
        print("┃ "+"~"*40)
    # Splitting for training and evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        n = 0
        for train_indices, test_indices in skf.split(x, y):    
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            n += 1
    # Training..
            training = time.time()
            model.fit(x=x_train, y=y_train, batch_size=96, epochs=20, verbose=0)
            print(f"┃ ⤷Fold_{n} training: {round(time.time() - training)} seconds..")
    # Evaluating..
            losses, accuracy = model.evaluate(x_test, y_test, verbose=0)
            print(f"┃ \t↺Fold_{n} evaluation: {losses=:.2f}, {accuracy=:.2f}")
    # Evaluating with a second dataset..
        print("┃ "+"~"*40)
        losses, accuracy = model.evaluate(xt, yt, verbose=0)
        print(f"┃ ↻End-of-batch evaluation: {losses=:.2f}, {accuracy=:.2f}")
    # Ending..
        print(f'┗━ FILE: {batch} time: {time.strftime("%H hours %M minutes %S seconds", time.gmtime(round(time.time() - batch_time)))}')
        if tmp_save:
            model.save(model_name)
            print(f"🗘 Model saved as: {model_name}")
        print("-"*40)
    # Summary time spent
    print(f'⇶ Full training took: {time.strftime("%H hours %M minutes %S seconds", time.gmtime(round(time.time() - start)))}')
    
    """ ~~~~ SAVE MODEL ~~~~ """
    model.save(model_name)
    print(f"⮔ Model saved as: {model_name}")


""" ~~~~ MAIN ~~~~ """
if __name__ == "__main__":
    main()