# Get data
scr
mkdir .cache
scp xl01339@131.227.70.182:/scratch/xl01339/03001627_2d_projection.zip /scratch/xl01339/
scp xl01339@131.227.70.182:/scratch/xl01339/miniconda3.zip /scratch/xl01339/
7z x miniconda3.zip
7z x 03001627_2d_projection.zip


# Create environment and run script
git clone https://github.com/yael-vinker/CLIPasso.git
git clone git@github.com:liangxg787/Enhancing_Sketch-to-3D_Controllability.git
cd Enhancing_Sketch-to-3D_Controllability
git checkout develop

cd ..
mkdir Enhancing_Sketch-to-3D_Controllability/output/2d_projection
cp 03001627_2d_projection/7/*.png ./Enhancing_Sketch-to-3D_Controllability/output/2d_projection/
rm CLIPasso/target_images/*
cp 03001627_2d_projection/7/*.png CLIPasso/target_images/
cd Enhancing_Sketch-to-3D_Controllability
export PYTHONPATH=$(pwd)

conda activate clip
cd ../CLIPasso
cd U2Net_/saved_models/
gdown "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&confirm=t"
cd ../../
python ../Enhancing_Sketch-to-3D_Controllability/src/dataset_preparation/sketches_processing/convert_to_sketch_with_CLIPasso.py
