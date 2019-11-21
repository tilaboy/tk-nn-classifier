# tk_nn_classifier
Detect whether the vacancy/job is from a direct employer or a recruitment agency

## Installation

    python setup.py develop

## Usage

'''
TRAIN: (example config_file can be found in cfg/)

tk-nn-train.py config_file

PROCESS BATCH:
tk-nn-infer.py model_dir trxml_data_path output_file


## TODOs:
- in config: set default option to reduce the needed parameter
- use csv or data_saver module to handle the output
