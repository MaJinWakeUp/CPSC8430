#!/bin/bash
var1="$1"
var2="$2"

# Call the Python script with the user inputs as arguments
python test.py --data_dir "$var1" --output "$var2"

python ./MLDS_hw2_1_data/bleu_eval.py "$var2"
