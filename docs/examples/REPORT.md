# Example reports

## Images: 224x224x3, 6 epochs per fold

> python ./run_training.py MODELbatches_6cat model_6cat_nasa

```
#################################################################
##                                                             ##
##                         INITIAL DATA                        ##
##                                                             ##
#################################################################


Model path:
        <CWD>/models/model_6cat_nasa

                      TRAIN_FILES       TEST_FILES
       train_dataset_labels_6c_01.npz   test_ns_nt_6c_01.npz
       train_dataset_labels_6c_02.npz   test_ns_nt_6c_02.npz
       train_dataset_labels_6c_03.npz   test_ns_nt_6c_03.npz
       train_dataset_labels_6c_04.npz   test_ns_nt_6c_04.npz
       train_dataset_labels_6c_05.npz   test_ns_nt_6c_05.npz
       train_dataset_labels_6c_06.npz   test_ns_nt_6c_06.npz
       train_dataset_labels_6c_07.npz
       train_dataset_labels_6c_08.npz
CLASSES:
      {'5': 0, '15': 1, '30': 2, '50': 3, '80': 4, '100': 5}
Checkpoint saves: True

#################################################################
##                                                             ##
##                            MODEL                            ##
##                                                             ##
#################################################################


Model: "Model_BIQA_NASA"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cv2d_64 (Conv2D)             (None, 222, 222, 64)      1792
_________________________________________________________________
mp2d_64 (MaxPooling2D)       (None, 111, 111, 64)      0
_________________________________________________________________
cv2d_128 (Conv2D)            (None, 109, 109, 128)     73856
_________________________________________________________________
mp2d_128 (MaxPooling2D)      (None, 54, 54, 128)       0
_________________________________________________________________
cv2d_256 (Conv2D)            (None, 52, 52, 256)       295168
_________________________________________________________________
mp2d_256 (MaxPooling2D)      (None, 26, 26, 256)       0
_________________________________________________________________
cv2d_512 (Conv2D)            (None, 24, 24, 512)       1180160
_________________________________________________________________
mp2d_512 (MaxPooling2D)      (None, 12, 12, 512)       0
_________________________________________________________________
cv2d_1024 (Conv2D)           (None, 10, 10, 1024)      4719616
_________________________________________________________________
mp2d_1024 (MaxPooling2D)     (None, 5, 5, 1024)        0
_________________________________________________________________
flatten (Flatten)            (None, 25600)             0
_________________________________________________________________
fc_4096 (Dense)              (None, 4096)              104861696
_________________________________________________________________
fc_128 (Dense)               (None, 128)               524416
_________________________________________________________________
fc_6 (Dense)                 (None, 6)                 774
=================================================================
Total params: 111,657,478
Trainable params: 111,657,478
Non-trainable params: 0
_________________________________________________________________
None

Press q to quit, Enter to continue..


#################################################################
##                                                             ##
##                           TRAINING                          ##
##                                                             ##
#################################################################


┏━ FILE: train_dataset_labels_6c_01.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 506 seconds..
┃       ↻Fold_1 training: fit_loss=1.79, fit_accuracy=0.16
┃       ↺Fold_1 evaluation: eval_loss=1.79, eval_accuracy=0.15
┃ ⤷Fold_2 training (6 epochs): 509 seconds..
┃       ↻Fold_2 training: fit_loss=1.68, fit_accuracy=0.25
┃       ↺Fold_2 evaluation: eval_loss=1.75, eval_accuracy=0.25
┃ ⤷Fold_3 training (6 epochs): 509 seconds..
┃       ↻Fold_3 training: fit_loss=1.52, fit_accuracy=0.34
┃       ↺Fold_3 evaluation: eval_loss=1.59, eval_accuracy=0.33
┃ ⤷Fold_4 training (6 epochs): 511 seconds..
┃       ↻Fold_4 training: fit_loss=1.22, fit_accuracy=0.49
┃       ↺Fold_4 evaluation: eval_loss=1.75, eval_accuracy=0.32
┃ ⤷Fold_5 training (6 epochs): 510 seconds..
┃       ↻Fold_5 training: fit_loss=1.04, fit_accuracy=0.57
┃       ↺Fold_5 evaluation: eval_loss=1.35, eval_accuracy=0.49
┗━ FILE: train_dataset_labels_6c_01.npz time: 00 h 42 min 56 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa00
=================================================================
┏━ FILE: train_dataset_labels_6c_02.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 505 seconds..
┃       ↻Fold_1 training: fit_loss=1.20, fit_accuracy=0.48
┃       ↺Fold_1 evaluation: eval_loss=1.59, eval_accuracy=0.30
┃ ⤷Fold_2 training (6 epochs): 509 seconds..
┃       ↻Fold_2 training: fit_loss=0.88, fit_accuracy=0.64
┃       ↺Fold_2 evaluation: eval_loss=1.63, eval_accuracy=0.39
┃ ⤷Fold_3 training (6 epochs): 548 seconds..
┃       ↻Fold_3 training: fit_loss=0.76, fit_accuracy=0.69
┃       ↺Fold_3 evaluation: eval_loss=1.06, eval_accuracy=0.60
┃ ⤷Fold_4 training (6 epochs): 561 seconds..
┃       ↻Fold_4 training: fit_loss=0.54, fit_accuracy=0.78
┃       ↺Fold_4 evaluation: eval_loss=0.97, eval_accuracy=0.62
┃ ⤷Fold_5 training (6 epochs): 558 seconds..
┃       ↻Fold_5 training: fit_loss=0.63, fit_accuracy=0.77
┃       ↺Fold_5 evaluation: eval_loss=0.70, eval_accuracy=0.72
┗━ FILE: train_dataset_labels_6c_02.npz time: 00 h 45 min 15 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa01
=================================================================
┏━ FILE: train_dataset_labels_6c_03.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 520 seconds..
┃       ↻Fold_1 training: fit_loss=1.08, fit_accuracy=0.56
┃       ↺Fold_1 evaluation: eval_loss=1.84, eval_accuracy=0.35
┃ ⤷Fold_2 training (6 epochs): 561 seconds..
┃       ↻Fold_2 training: fit_loss=0.62, fit_accuracy=0.76
┃       ↺Fold_2 evaluation: eval_loss=1.64, eval_accuracy=0.49
┃ ⤷Fold_3 training (6 epochs): 564 seconds..
┃       ↻Fold_3 training: fit_loss=0.39, fit_accuracy=0.87
┃       ↺Fold_3 evaluation: eval_loss=0.93, eval_accuracy=0.68
┃ ⤷Fold_4 training (6 epochs): 567 seconds..
┃       ↻Fold_4 training: fit_loss=0.22, fit_accuracy=0.92
┃       ↺Fold_4 evaluation: eval_loss=0.65, eval_accuracy=0.83
┃ ⤷Fold_5 training (6 epochs): 578 seconds..
┃       ↻Fold_5 training: fit_loss=0.16, fit_accuracy=0.94
┃       ↺Fold_5 evaluation: eval_loss=0.40, eval_accuracy=0.87
┗━ FILE: train_dataset_labels_6c_03.npz time: 00 h 47 min 06 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa02
=================================================================
┏━ FILE: train_dataset_labels_6c_04.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 605 seconds..
┃       ↻Fold_1 training: fit_loss=0.96, fit_accuracy=0.61
┃       ↺Fold_1 evaluation: eval_loss=1.78, eval_accuracy=0.38
┃ ⤷Fold_2 training (6 epochs): 566 seconds..
┃       ↻Fold_2 training: fit_loss=0.54, fit_accuracy=0.80
┃       ↺Fold_2 evaluation: eval_loss=1.34, eval_accuracy=0.56
┃ ⤷Fold_3 training (6 epochs): 540 seconds..
┃       ↻Fold_3 training: fit_loss=0.29, fit_accuracy=0.90
┃       ↺Fold_3 evaluation: eval_loss=0.83, eval_accuracy=0.75
┃ ⤷Fold_4 training (6 epochs): 536 seconds..
┃       ↻Fold_4 training: fit_loss=0.09, fit_accuracy=0.97
┃       ↺Fold_4 evaluation: eval_loss=0.26, eval_accuracy=0.88
┃ ⤷Fold_5 training (6 epochs): 557 seconds..
┃       ↻Fold_5 training: fit_loss=0.18, fit_accuracy=0.94
┃       ↺Fold_5 evaluation: eval_loss=0.24, eval_accuracy=0.94
┗━ FILE: train_dataset_labels_6c_04.npz time: 00 h 47 min 18 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa03
=================================================================
┏━ FILE: train_dataset_labels_6c_05.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 587 seconds..
┃       ↻Fold_1 training: fit_loss=0.61, fit_accuracy=0.76
┃       ↺Fold_1 evaluation: eval_loss=1.53, eval_accuracy=0.50
┃ ⤷Fold_2 training (6 epochs): 710 seconds..
┃       ↻Fold_2 training: fit_loss=0.15, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.65, eval_accuracy=0.78
┃ ⤷Fold_3 training (6 epochs): 690 seconds..
┃       ↻Fold_3 training: fit_loss=0.08, fit_accuracy=0.98
┃       ↺Fold_3 evaluation: eval_loss=0.27, eval_accuracy=0.93
┃ ⤷Fold_4 training (6 epochs): 926 seconds..
┃       ↻Fold_4 training: fit_loss=0.03, fit_accuracy=0.99
┃       ↺Fold_4 evaluation: eval_loss=0.16, eval_accuracy=0.96
┃ ⤷Fold_5 training (6 epochs): 599 seconds..
┃       ↻Fold_5 training: fit_loss=0.01, fit_accuracy=1.00
┃       ↺Fold_5 evaluation: eval_loss=0.06, eval_accuracy=0.98
┗━ FILE: train_dataset_labels_6c_05.npz time: 00 h 59 min 19 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa04
=================================================================
┏━ FILE: train_dataset_labels_6c_06.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 617 seconds..
┃       ↻Fold_1 training: fit_loss=0.20, fit_accuracy=0.92
┃       ↺Fold_1 evaluation: eval_loss=0.54, eval_accuracy=0.79
┃ ⤷Fold_2 training (6 epochs): 567 seconds..
┃       ↻Fold_2 training: fit_loss=0.38, fit_accuracy=0.88
┃       ↺Fold_2 evaluation: eval_loss=0.50, eval_accuracy=0.80
┃ ⤷Fold_3 training (6 epochs): 582 seconds..
┃       ↻Fold_3 training: fit_loss=0.12, fit_accuracy=0.97
┃       ↺Fold_3 evaluation: eval_loss=0.11, eval_accuracy=0.97
┃ ⤷Fold_4 training (6 epochs): 605 seconds..
┃       ↻Fold_4 training: fit_loss=0.01, fit_accuracy=1.00
┃       ↺Fold_4 evaluation: eval_loss=0.02, eval_accuracy=1.00
┃ ⤷Fold_5 training (6 epochs): 599 seconds..
┃       ↻Fold_5 training: fit_loss=0.00, fit_accuracy=1.00
┃       ↺Fold_5 evaluation: eval_loss=0.03, eval_accuracy=1.00
┗━ FILE: train_dataset_labels_6c_06.npz time: 00 h 50 min 11 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa05
=================================================================
┏━ FILE: train_dataset_labels_6c_07.npz :
┃ train_set:    (1200, 224, 224, 3) » train_labels:     (1200,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 604 seconds..
┃       ↻Fold_1 training: fit_loss=0.44, fit_accuracy=0.82
┃       ↺Fold_1 evaluation: eval_loss=0.80, eval_accuracy=0.65
┃ ⤷Fold_2 training (6 epochs): 606 seconds..
┃       ↻Fold_2 training: fit_loss=0.02, fit_accuracy=1.00
┃       ↺Fold_2 evaluation: eval_loss=0.08, eval_accuracy=0.98
┃ ⤷Fold_3 training (6 epochs): 596 seconds..
┃       ↻Fold_3 training: fit_loss=0.03, fit_accuracy=0.99
┃       ↺Fold_3 evaluation: eval_loss=0.03, eval_accuracy=0.98
┃ ⤷Fold_4 training (6 epochs): 597 seconds..
┃       ↻Fold_4 training: fit_loss=0.01, fit_accuracy=1.00
┃       ↺Fold_4 evaluation: eval_loss=0.01, eval_accuracy=1.00
┃ ⤷Fold_5 training (6 epochs): 617 seconds..
┃       ↻Fold_5 training: fit_loss=0.00, fit_accuracy=1.00
┃       ↺Fold_5 evaluation: eval_loss=0.00, eval_accuracy=1.00
┗━ FILE: train_dataset_labels_6c_07.npz time: 00 h 51 min 02 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa06
=================================================================
┏━ FILE: train_dataset_labels_6c_08.npz :
┃ train_set:    (666, 224, 224, 3) » train_labels:      (666,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (6 epochs): 340 seconds..
┃       ↻Fold_1 training: fit_loss=0.04, fit_accuracy=0.99
┃       ↺Fold_1 evaluation: eval_loss=0.26, eval_accuracy=0.90
┃ ⤷Fold_2 training (6 epochs): 317 seconds..
┃       ↻Fold_2 training: fit_loss=0.45, fit_accuracy=0.85
┃       ↺Fold_2 evaluation: eval_loss=0.80, eval_accuracy=0.75
┃ ⤷Fold_3 training (6 epochs): 327 seconds..
┃       ↻Fold_3 training: fit_loss=0.02, fit_accuracy=1.00
┃       ↺Fold_3 evaluation: eval_loss=0.04, eval_accuracy=0.98
┃ ⤷Fold_4 training (6 epochs): 302 seconds..
┃       ↻Fold_4 training: fit_loss=0.00, fit_accuracy=1.00
┃       ↺Fold_4 evaluation: eval_loss=0.00, eval_accuracy=1.00
┃ ⤷Fold_5 training (6 epochs): 357 seconds..
┃       ↻Fold_5 training: fit_loss=0.00, fit_accuracy=1.00
┃       ↺Fold_5 evaluation: eval_loss=0.00, eval_accuracy=1.00
┗━ FILE: train_dataset_labels_6c_08.npz time: 00 h 27 min 46 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa07
=================================================================
⇶ Full training took: 06 h 15 min 01 sec
⮔ Model saved as:
        <CWD>/models/model_6cat_nasa

⮔ Train data saved as:
        <CWD>/MODELbatches_6cat/model_6cat_nasa_train_stats.json
        <CWD>/MODELbatches_6cat/model_6cat_nasa_train_history.json

#################################################################
##                                                             ##
##                           TESTING                           ##
##                                                             ##
#################################################################


⇊ Adding Softmax Layer to model
⮔ Statistics saved as:
        <CWD>/MODELbatches_6cat/model_6cat_nasa_stats.npz
```

## Images: 28x28x3, 20 epochs per fold

> python ./run_training.py ./MODELbatches_6cat_frag model_6cat_nasa_frag

```
#################################################################
##                                                             ##
##                         INITIAL DATA                        ##
##                                                             ##
#################################################################


Model path:
        <CWD>/models/model_6cat_nasa_frag

                      TRAIN_FILES       TEST_FILES
       train_dataset_labels_6c_01.npz   test_ns_nt_6c_01.npz
       train_dataset_labels_6c_02.npz   test_ns_nt_6c_02.npz
       train_dataset_labels_6c_03.npz   test_ns_nt_6c_03.npz
       train_dataset_labels_6c_04.npz   test_ns_nt_6c_04.npz
       train_dataset_labels_6c_05.npz   test_ns_nt_6c_05.npz
       train_dataset_labels_6c_06.npz   test_ns_nt_6c_06.npz
       train_dataset_labels_6c_07.npz
       train_dataset_labels_6c_08.npz
CLASSES:
      {'5': 0, '15': 1, '30': 2, '50': 3, '80': 4, '100': 5}
Checkpoint saves: True

#################################################################
##                                                             ##
##                            MODEL                            ##
##                                                             ##
#################################################################


Model: "Model_BIQA_NASA"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cv2d_64 (Conv2D)             (None, 26, 26, 64)        1792
_________________________________________________________________
mp2d_64 (MaxPooling2D)       (None, 13, 13, 64)        0
_________________________________________________________________
cv2d_128 (Conv2D)            (None, 11, 11, 128)       73856
_________________________________________________________________
mp2d_128 (MaxPooling2D)      (None, 5, 5, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 3200)              0
_________________________________________________________________
fc_128 (Dense)               (None, 128)               409728
_________________________________________________________________
fc_6 (Dense)                 (None, 6)                 774
=================================================================
Total params: 486,150
Trainable params: 486,150
Non-trainable params: 0
_________________________________________________________________
None

Press q to quit, Enter to continue..


#################################################################
##                                                             ##
##                           TRAINING                          ##
##                                                             ##
#################################################################


┏━ FILE: train_dataset_labels_6c_01.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 644 seconds..
┃       ↻Fold_1 training: fit_loss=0.42, fit_accuracy=0.83
┃       ↺Fold_1 evaluation: eval_loss=0.50, eval_accuracy=0.80
┃ ⤷Fold_2 training (20 epochs): 643 seconds..
┃       ↻Fold_2 training: fit_loss=0.26, fit_accuracy=0.89
┃       ↺Fold_2 evaluation: eval_loss=0.45, eval_accuracy=0.83
┃ ⤷Fold_3 training (20 epochs): 667 seconds..
┃       ↻Fold_3 training: fit_loss=0.20, fit_accuracy=0.92
┃       ↺Fold_3 evaluation: eval_loss=0.29, eval_accuracy=0.88
┃ ⤷Fold_4 training (20 epochs): 642 seconds..
┃       ↻Fold_4 training: fit_loss=0.17, fit_accuracy=0.93
┃       ↺Fold_4 evaluation: eval_loss=0.25, eval_accuracy=0.90
┃ ⤷Fold_5 training (20 epochs): 648 seconds..
┃       ↻Fold_5 training: fit_loss=0.15, fit_accuracy=0.94
┃       ↺Fold_5 evaluation: eval_loss=0.20, eval_accuracy=0.92
┗━ FILE: train_dataset_labels_6c_01.npz time: 00 h 54 min 22 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag00
=================================================================
┏━ FILE: train_dataset_labels_6c_02.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 641 seconds..
┃       ↻Fold_1 training: fit_loss=0.15, fit_accuracy=0.94
┃       ↺Fold_1 evaluation: eval_loss=0.51, eval_accuracy=0.85
┃ ⤷Fold_2 training (20 epochs): 642 seconds..
┃       ↻Fold_2 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.23, eval_accuracy=0.91
┃ ⤷Fold_3 training (20 epochs): 641 seconds..
┃       ↻Fold_3 training: fit_loss=0.12, fit_accuracy=0.95
┃       ↺Fold_3 evaluation: eval_loss=0.17, eval_accuracy=0.93
┃ ⤷Fold_4 training (20 epochs): 650 seconds..
┃       ↻Fold_4 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_4 evaluation: eval_loss=0.13, eval_accuracy=0.95
┃ ⤷Fold_5 training (20 epochs): 637 seconds..
┃       ↻Fold_5 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_5 evaluation: eval_loss=0.12, eval_accuracy=0.95
┗━ FILE: train_dataset_labels_6c_02.npz time: 00 h 53 min 48 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag01
=================================================================
┏━ FILE: train_dataset_labels_6c_03.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 638 seconds..
┃       ↻Fold_1 training: fit_loss=0.13, fit_accuracy=0.94
┃       ↺Fold_1 evaluation: eval_loss=0.46, eval_accuracy=0.88
┃ ⤷Fold_2 training (20 epochs): 639 seconds..
┃       ↻Fold_2 training: fit_loss=0.12, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.18, eval_accuracy=0.93
┃ ⤷Fold_3 training (20 epochs): 630 seconds..
┃       ↻Fold_3 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_3 evaluation: eval_loss=0.12, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 627 seconds..
┃       ↻Fold_4 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_4 evaluation: eval_loss=0.12, eval_accuracy=0.95
┃ ⤷Fold_5 training (20 epochs): 625 seconds..
┃       ↻Fold_5 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_5 evaluation: eval_loss=0.11, eval_accuracy=0.95
┗━ FILE: train_dataset_labels_6c_03.npz time: 00 h 52 min 55 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag02
=================================================================
┏━ FILE: train_dataset_labels_6c_04.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 621 seconds..
┃       ↻Fold_1 training: fit_loss=0.13, fit_accuracy=0.95
┃       ↺Fold_1 evaluation: eval_loss=0.39, eval_accuracy=0.88
┃ ⤷Fold_2 training (20 epochs): 624 seconds..
┃       ↻Fold_2 training: fit_loss=0.10, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.20, eval_accuracy=0.93
┃ ⤷Fold_3 training (20 epochs): 619 seconds..
┃       ↻Fold_3 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_3 evaluation: eval_loss=0.12, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 616 seconds..
┃       ↻Fold_4 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_4 evaluation: eval_loss=0.13, eval_accuracy=0.95
┃ ⤷Fold_5 training (20 epochs): 613 seconds..
┃       ↻Fold_5 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_5 evaluation: eval_loss=0.09, eval_accuracy=0.96
┗━ FILE: train_dataset_labels_6c_04.npz time: 00 h 51 min 50 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag03
=================================================================
┏━ FILE: train_dataset_labels_6c_05.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 613 seconds..
┃       ↻Fold_1 training: fit_loss=0.12, fit_accuracy=0.95
┃       ↺Fold_1 evaluation: eval_loss=0.38, eval_accuracy=0.89
┃ ⤷Fold_2 training (20 epochs): 614 seconds..
┃       ↻Fold_2 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.22, eval_accuracy=0.92
┃ ⤷Fold_3 training (20 epochs): 615 seconds..
┃       ↻Fold_3 training: fit_loss=0.09, fit_accuracy=0.96
┃       ↺Fold_3 evaluation: eval_loss=0.12, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 616 seconds..
┃       ↻Fold_4 training: fit_loss=0.09, fit_accuracy=0.96
┃       ↺Fold_4 evaluation: eval_loss=0.11, eval_accuracy=0.96
┃ ⤷Fold_5 training (20 epochs): 619 seconds..
┃       ↻Fold_5 training: fit_loss=0.11, fit_accuracy=0.96
┃       ↺Fold_5 evaluation: eval_loss=0.09, eval_accuracy=0.96
┗━ FILE: train_dataset_labels_6c_05.npz time: 00 h 51 min 33 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag04
=================================================================
┏━ FILE: train_dataset_labels_6c_06.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 622 seconds..
┃       ↻Fold_1 training: fit_loss=0.13, fit_accuracy=0.94
┃       ↺Fold_1 evaluation: eval_loss=0.35, eval_accuracy=0.89
┃ ⤷Fold_2 training (20 epochs): 622 seconds..
┃       ↻Fold_2 training: fit_loss=0.10, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.21, eval_accuracy=0.92
┃ ⤷Fold_3 training (20 epochs): 619 seconds..
┃       ↻Fold_3 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_3 evaluation: eval_loss=0.11, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 620 seconds..
┃       ↻Fold_4 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_4 evaluation: eval_loss=0.13, eval_accuracy=0.95
┃ ⤷Fold_5 training (20 epochs): 623 seconds..
┃       ↻Fold_5 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_5 evaluation: eval_loss=0.17, eval_accuracy=0.93
┗━ FILE: train_dataset_labels_6c_06.npz time: 00 h 52 min 02 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag05
=================================================================
┏━ FILE: train_dataset_labels_6c_07.npz :
┃ train_set:    (76800, 28, 28, 3) » train_labels:      (76800,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 629 seconds..
┃       ↻Fold_1 training: fit_loss=0.13, fit_accuracy=0.94
┃       ↺Fold_1 evaluation: eval_loss=0.35, eval_accuracy=0.90
┃ ⤷Fold_2 training (20 epochs): 623 seconds..
┃       ↻Fold_2 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.26, eval_accuracy=0.91
┃ ⤷Fold_3 training (20 epochs): 619 seconds..
┃       ↻Fold_3 training: fit_loss=0.11, fit_accuracy=0.95
┃       ↺Fold_3 evaluation: eval_loss=0.13, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 622 seconds..
┃       ↻Fold_4 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_4 evaluation: eval_loss=0.10, eval_accuracy=0.96
┃ ⤷Fold_5 training (20 epochs): 622 seconds..
┃       ↻Fold_5 training: fit_loss=0.10, fit_accuracy=0.96
┃       ↺Fold_5 evaluation: eval_loss=0.11, eval_accuracy=0.95
┗━ FILE: train_dataset_labels_6c_07.npz time: 00 h 52 min 10 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag06
=================================================================
┏━ FILE: train_dataset_labels_6c_08.npz :
┃ train_set:    (42624, 28, 28, 3) » train_labels:      (42624,)
┃ _______________________________________________________________
┃ ⤷Fold_1 training (20 epochs): 346 seconds..
┃       ↻Fold_1 training: fit_loss=0.13, fit_accuracy=0.94
┃       ↺Fold_1 evaluation: eval_loss=0.46, eval_accuracy=0.89
┃ ⤷Fold_2 training (20 epochs): 344 seconds..
┃       ↻Fold_2 training: fit_loss=0.12, fit_accuracy=0.95
┃       ↺Fold_2 evaluation: eval_loss=0.17, eval_accuracy=0.93
┃ ⤷Fold_3 training (20 epochs): 345 seconds..
┃       ↻Fold_3 training: fit_loss=0.13, fit_accuracy=0.94
┃       ↺Fold_3 evaluation: eval_loss=0.12, eval_accuracy=0.95
┃ ⤷Fold_4 training (20 epochs): 344 seconds..
┃       ↻Fold_4 training: fit_loss=0.09, fit_accuracy=0.96
┃       ↺Fold_4 evaluation: eval_loss=0.14, eval_accuracy=0.94
┃ ⤷Fold_5 training (20 epochs): 343 seconds..
┃       ↻Fold_5 training: fit_loss=0.09, fit_accuracy=0.96
┃       ↺Fold_5 evaluation: eval_loss=0.10, eval_accuracy=0.95
┗━ FILE: train_dataset_labels_6c_08.npz time: 00 h 28 min 52 sec
🗘 Model's checkpoint saved as:
        <CWD>/models/model_6cat_nasa_frag07
=================================================================
⇶ Full training took: 06 h 37 min 42 sec
⮔ Model saved as:
        <CWD>/models/model_6cat_nasa_frag

⮔ Train data saved as:
        <CWD>/MODELbatches_6cat_frag/model_6cat_nasa_frag_train_stats.json
        <CWD>/MODELbatches_6cat_frag/model_6cat_nasa_frag_train_history.json

#################################################################
##                                                             ##
##                           TESTING                           ##
##                                                             ##
#################################################################


⇊ Adding Softmax Layer to model
⮔ Statistics saved as:
        <CWD>/MODELbatches_6cat_frag/model_6cat_nasa_frag_stats.npz
```

<br>

---

`<CWD>` -  states for current working directory