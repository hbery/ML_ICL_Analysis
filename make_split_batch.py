"""
:Author: Adam Twardosz (hbery@github.com)
"""

""" IMPORTS """
import os
import json
from make_batch import (
    prebatch_files,
    optimize_batch_size,
    calculate_mixed_batch_and_save
)


def make_mixed_batches(base_path: str):
    """Prepare batches for model training

    :param base_path: [description]
    :type base_path: str
    """
    num_classes = 6
    
    # setting directories
    nasa = os.path.abspath(f"./NASA_224x224_{num_classes}_comp_levels")
    nature = os.path.abspath(f"./NATURE_224x224_{num_classes}_comp_levels")
    nsfiles = []
    ntfiles = []
    
    out1_name = f"./outNsNt_{num_classes}cat"
    out2_name = f"./outNtNs_{num_classes}cat"
    out_ns_nt, out_nt_ns = None, None
    if os.path.isdir(os.path.abspath(out1_name)):
        out_ns_nt = os.path.abspath(out1_name)
    else:
        os.mkdir(os.path.join(base_path, out1_name))
        out_ns_nt = os.path.abspath(out1_name)

    if os.path.isdir(os.path.abspath(out2_name)):
        out_nt_ns = os.path.abspath(out2_name)
    else:
        os.mkdir(os.path.join(base_path, out2_name))
        out_nt_ns = os.path.abspath(out2_name)

    # get all files into one list from each set
    class_list = os.listdir(nasa)
    [ nsfiles.append(os.listdir(os.path.join(nasa, directory))) for directory in os.listdir(nasa) ]
    [ ntfiles.append(os.listdir(os.path.join(nature, directory))) for directory in os.listdir(nature) ]

    # saving classes
    classes = {}
    for idx, label in enumerate(sorted(list(set(class_list)), key=lambda x: int(x))):
        classes[label] = idx

    with open(os.path.join(out_ns_nt, f"classes_{num_classes}cat.json"), "w") as fclass:
        json.dump(classes, fclass)
            
    with open(os.path.join(out_nt_ns, f"classes_{num_classes}cat.json"), "w") as fclass:
        json.dump(classes, fclass)
    
    # We do not care about residual batches
    ( nsbatches, _ ) = prebatch_files(nsfiles, 200)
    ( ntbatches, _ ) = prebatch_files(ntfiles, 200)
    
    num_batches = len(ntbatches) if len(ntbatches) < len(nsbatches) else len(nsbatches)
    
    # calculate and save    
    for bnum in range(num_batches):
        # out_ns_nt
        calculate_mixed_batch_and_save(
            nasa,
            nature,
            nsbatches[bnum],
            ntbatches[bnum],
            os.path.join(
                out_ns_nt,
                f'nstrain_nttest_{num_classes}c_{(bnum+1):02}.npz'),
            classes
        )
        
        # out_nt_ns
        calculate_mixed_batch_and_save(
            nature,
            nasa,
            ntbatches[bnum],
            nsbatches[bnum],
            os.path.join(
                out_nt_ns,
                f'nttrain_nstest_{num_classes}c_{(bnum+1):02}.npz'),
            classes
        )


""" MAIN """
if __name__ == "__main__":
    make_mixed_batches(os.getcwd())