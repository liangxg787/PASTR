#!/bin/bash

conda activate clip
export PYTHONPATH=$(pwd)
cd ../CLIPasso
python ../Enhancing_Sketch-to-3D_Controllability/src/dataset_preparation/sketches_processing/convert_to_sketch_with_CLIPasso.py