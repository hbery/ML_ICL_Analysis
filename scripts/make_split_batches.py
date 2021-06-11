#!/usr/bin/env python3

"""Make .npz batches from two dataset to make 'train_' and 'test_' files
    to ease the training and testing

    :Date: 05.2021
    :Author: Adam Twardosz (a.twardosz98@gmail.com, https://github.com/hbery)
"""

import os
import json
from utils import (
    prebatch_files,
    calculate_batch_and_save,
    calculate_mixed_batch_and_save,
    pop_testing_set
)


def make_mixed_batches(base_path: str):
    """Prepare batches for model training

    :param base_path: Path to where are folders with data
    :type base_path: str
    """
    num_classes = 6
    
    # setting directories
    nasa = os.path.abspath(f"./NASA_224x224_{num_classes}_comp_levels")
    nature = os.path.abspath(f"./NATURE_224x224_{num_classes}_comp_levels")
    nsfiles = []
    test_nt = []
    
    out_name = f"./MODELbatches_{num_classes}cat"
    out_data = None
    if os.path.isdir(os.path.abspath(out_name)):
        out_data = os.path.abspath(out_name)
    else:
        os.mkdir(os.path.join(base_path, out_name))
        out_data = os.path.abspath(out_name)

    # get all files into one list from each set
    class_list = os.listdir(nasa)
    [ nsfiles.append(os.listdir(os.path.join(nasa, directory))) for directory in os.listdir(nasa) ]
    [ test_nt.append(os.listdir(os.path.join(nature, directory))) for directory in os.listdir(nature) ]

    # saving classes
    classes = {}
    for idx, label in enumerate(sorted(list(set(class_list)), key=lambda x: int(x))):
        classes[label] = idx

    with open(os.path.join(out_data, f"classes_{num_classes}cat.json"), "w") as fclass:
        json.dump(classes, fclass)
    
    (test_ns, train_ns) = pop_testing_set(all_files=nsfiles, batch_k=len(test_nt[0]))
    
    # prebatch train_files
    ( batches_train, res_train ) = prebatch_files(train_ns, 200)
    
    # prebatch test_files
    ( batches_ns_test, res_ns_test ) = prebatch_files(test_ns, 200)
    ( batches_nt_test, res_nt_test ) = prebatch_files(test_nt, 200)
    
    # calculate and save TRAIN SET    
    for bnum, batch in enumerate(batches_train):
        calculate_batch_and_save( nasa, batch,
            os.path.join( out_data,
                f'train_dataset_labels_{num_classes}c_{(bnum+1):02}.npz'),
            classes,
            0
        )

    calculate_batch_and_save( nasa, res_train,
            os.path.join( out_data,
                f'train_dataset_labels_{num_classes}c_{(len(batches_train)+1):02}.npz'),
            classes,
            0
        )
    
    # calculate and save TEST SETS
    for bnum in range(len(batches_ns_test)):
        calculate_mixed_batch_and_save( nasa, nature, batches_ns_test[bnum], batches_nt_test[bnum],
            os.path.join( out_data,
                f'test_ns_nt_{num_classes}c_{(bnum+1):02}.npz'),
            classes,
            0
        )

    calculate_mixed_batch_and_save( nasa, nature, res_ns_test, res_nt_test,
            os.path.join( out_data,
                f'test_ns_nt_{num_classes}c_{(len(batches_ns_test)+1):02}.npz'),
            classes,
            0
        )


""" MAIN """
if __name__ == "__main__":
    make_mixed_batches(os.getcwd())