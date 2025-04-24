# Learning Structure-from-Motion with Graph Attention Networks  
**Paper | arXiv**

This is the implementation of the **Graph Attention Structure-from-Motion (GASFM)** architecture, presented in our **CVPR 2024** paper *Learning Structure-from-Motion with Graph Attention Networks*. The codebase is forked from the implementation of the ICCV 2021 paper *Deep Permutation Equivariant Structure from Motion*, available at:  
👉 [https://github.com/drormoran/Equivariant-SFM](https://github.com/drormoran/Equivariant-SFM)  
That architecture is also used as a baseline and referred to as **DPESFM** in our paper.


The primary focus of our paper is on **Euclidean reconstruction of novel test scenes**, achieved by training a graph attention network on a few scenes, which then generalizes well enough to provide an initial solution, locally refined by **bundle adjustment**.  
Our experiments demonstrate that high-quality reconstructions can be acquired around **5–10× faster than COLMAP**.


## 📁 Contents
- [Setup](#setup)
- [Usage](#usage)
- [Citation](#citation)


## ⚙️ Setup

This repository is implemented with **Python 3.9**, and in order to run **bundle adjustment**, it requires **Linux** (e.g., Ubuntu 22.04).  
You should also have a **CUDA-capable GPU**.

### 📂 Directory Structure

```bash
gasfm
├── bundle_adjustment
├── code
├── datasets
├── environment.yml
```


### 🐍 Conda Environment

First create and activate a conda environment:

```bash
conda create -n gasfm
conda activate gasfm
conda install python=3.9
```

Install the Facebook research vision core library:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore
```

Install PyTorch and CUDA toolkit 11.6:

```bash
conda install -c conda-forge cudatoolkit-dev=11.6
conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install TensorBoard:

```bash
pip install future tb-nightly
```

Install all remaining dependencies:

```bash
conda env update -n gasfm -f environment.yml
```

### ✅ Verify Installations

Check if PyTorch has CUDA:

```bash
python -c 'import torch; assert torch.cuda.is_available()'
```

Check if PyTorch Geometric is installed:

```bash
python -c 'import torch_geometric'
```

---

### 🔧 PyCeres

Follow the bundle adjustment instructions provided in the repo.

---

## 📦 Data and Pretrained Models

Find datasets and pretrained models attached to the [GitHub release](https://github.com/lucasbrynte/gasfm/releases).

### 📁 Datasets

Create dataset directory and download:

```bash
mkdir -p datasets
cd datasets
wget https://github.com/lucasbrynte/gasfm/releases/download/data/datasets.zip
unzip datasets.zip
```

### Pretrained Models

```bash
cd ..
mkdir -p pretrained_models
cd pretrained_models

wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_noaug.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_rhaug-15-20.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_euc_rhaug-15-20_outliers0.1.pt
wget https://github.com/lucasbrynte/gasfm/releases/download/data/gasfm_proj_noaug.pt
```

---

## 🚀 Usage

### 🧪 Novel Scene Reconstruction

Navigate to the `code` subdirectory. Activate your environment.

Train a model from scratch:

```bash
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning
```

Disable fine-tuning:

```bash
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning --skip-fine-tuning
```

Skip training and use a pretrained model:

```bash
python main.py --conf path/to/conf --exp-dir exp/output/path --pretrained-model-path /abs/path/to/pretrained/model multi-scene-learning --skip-training
```

Or using existing experiment directory:

```bash
python main.py --conf path/to/conf --exp-dir exp/output/path multi-scene-learning --skip-training --old-exp-dir /another/path --pretrained-model-filename best_model.pt
```

Override config values from command line:

```bash
python main.py --conf path/to/conf --external_params train.n_epochs=100000 eval.eval_interval=100 --exp-dir exp/output/path multi-scene-learning
```

---

### 🗺️ Single-Scene Recovery

Run single-scene optimization (example: `AlcatrazCourtyard`):

```bash
python main.py --conf path/to/conf --exp-dir exp/output/path single-scene-optim --scene-name-exp-subdir --scene AlcatrazCourtyard
```

With override:

```bash
python main.py --conf path/to/conf --external_params train.n_epochs=100000 eval.eval_interval=100 --exp-dir exp/output/path single-scene-optim --scene-name-exp-subdir --scene AlcatrazCourtyard
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Brynte_2024_CVPR,
    author    = {Brynte, Lucas and Iglesias, Jos\'e Pedro and Olsson, Carl and Kahl, Fredrik},
    title     = {Learning Structure-from-Motion with Graph Attention Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4808-4817}
}
```

---

