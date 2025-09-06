# PASTR: Enhancing Controllability of Part-aware Sketch-to-3D Generation via Tiered Rectified Flow

## 0 Environment
You must make sure the GCC version >= 9.0.0
```shell
conda env create -f environment.yml
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 1 Dataset

### 1.0 Update submodule
```
git submodule init
git submodule update --init --recursive --remote

# If updateing error:
git submodule update --force --recursive --init --remote
```

### 1.1 3D Dataset

#### 1.1.1 Manifold
Reference: https://github.com/hjwdzh/Manifold
##### a. Compile Manifold
```
cd external/Manifold
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
##### b. Convert to Manifold data
```
python src/dataset_preparation/shapeNet_processing/convert_to_manifold.py
```

#### 1.1.2 ManifoldPlus
Reference:https://github.com/hjwdzh/ManifoldPlus.git
##### a. Compile ManifoldPlus
```
cd external/ManifoldPlus
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```
##### b. Convert to ManifoldPlus data
```
python src/dataset_preparation/shapeNet_processing/convert_to_manifold.py
```

#### 1.1.2 SPAGHETTI representation
Reference: https://github.com/liangxg787/spaghetti
```
git submodule update
cd external/spaghetti
python setup.py sdist
pip install dist/spaghetti-1.0.0.tar.gz

sh make_dataset.sh
```

#### 1.1.3 Building SPAGHETTI environment (Optional)
Reference: https://github.com/amirhertz/spaghetti
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-image=0.18.1 igl -c conda-forge
pip install vtk==9.2.4 tqdm pynput requests
```

### 1.2 Sketch Dataset

#### 1.2.1 2D projection
```
python src/dataset_preparation/sketches_processing/2D_projection.py
```

#### 1.2.2 informative-drawings
Reference: https://github.com/carolineec/informative-drawings.git

**a. Installation**
```
git clone https://github.com/carolineec/informative-drawings.git
cd informative-drawings

conda env create -f environment.yml
conda activate drawings

conda activate drawings
pip install git+https://github.com/openai/CLIP.git
```

**b. Convert to sketches**
```
sh sh_script/convert_to_sketch_with_Informative.sh
```

#### 1.2.3 CLIPasso
Reference: https://github.com/yael-vinker/CLIPasso.git
**a. Installation**
```
# Check the gcc version
gcc -v
# The gcc version must be 9.2.0, 9.3.0, 9.4.0, or 9.5.0

mamba create -n clipasso python=3.7
mamba install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
mamba install -y numpy scikit-image
mamba install -y -c anaconda cmake
mamba install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils numba torch-tools visdom IPython wandb ipywidgets ftfy regex tqdm scipy==1.6.2
pip install git+https://github.com/openai/CLIP.git

git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
DIFFVG_CUDA=1 python setup.py install

git clone https://github.com/yael-vinker/CLIPasso.git
cd CLIPasso

# (TypeError: Descriptors cannot not be created directly)
pip install protobuf==3.20.*

cd U2Net_/saved_models/
gdown "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&confirm=t"
```

**b. Convert to sketches**
```
conda activate clipasso
cd ../CLIPasso
python ../Enhancing_Sketch-to-3D_Controllability/src/dataset_preparation/sketches_processing/convert_to_sketch.py
```


## 2 Metrics
### 2.1 Chamfer Distance (CD)

### 2.2 Earth Mover's Distance (EMD)

