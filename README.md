# ML_ICL_Analysis

## About Project

This project treats about JPEG Compression Level Detection using Machine Learning and CNN Tensorflow model. This repository contains all things that we used to build a model and test it.

## About Repository

This Git repository contains useful scripts and information that can be helpful during project reproduction. Scripts are contained in [Scripts](./scripts) folder. At first we have done it in Google Colab, so in [Notebooks](./notebooks) folder are notebooks that we used. Documentation about image preparation can be found in [Docs](./docs). Examples of model training and statistics/charts produced by scripts is located in [Examples](./docs/examples) folder.

## Tree view of our project

```bash
├── charts
│   └── < .png charts produced by make_charts.py >
├── MODELbatches_6cat
│   └── < .npz batches >
├── MODELbatches_6cat_frag
│   └── < .npz batches >
├── models
│   ├── model_6cat_nasa
│   │   └── <...>
│   └── model_6cat_nasa_frag
│       └── <...>
├── NASA_224x224_6_comp_levels
│   └── < .jpg images >
├── NATURE_224x224_6_comp_levels
│   └── < .jpg images >
├── statistics
│   ├── classes_6cat.json
│   └── < statistics files produced by run_training.py and/or only_testing.py >
├── images2classfolders.py
├── make_batches.py
├── make_charts.py
├── make_split_batches.py
├── model.py
├── model_frag.py
├── only_testing.py
├── requirements.txt
├── run_training.py
└── utils.py
```
