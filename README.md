# SIMORGH

![Ensemble Aggregation](Birds.gif)

## Table of Contents: 
- [The workflow](#The-workflow)
- [Installation](#Installation)
- [Training the model](#Training-the-model)
- [Running inference](#Running-inference)
- [License](#License)
- [Acknowledgement](#Acknowledgement)
- [Citation](#Citation)


## **Installation**
To set up the conda environment:

```bash
conda create -n dl python=3.10 -y
conda activate dl

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install e3nn glob2 tqdm h5py numpy scipy wandb scikit-learn
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
