# SIMORGH

![Ensemble Aggregation](Birds.gif)

## Table of Contents: 
- [The workflow](#The-workflow)
- [Installation](#Installation)
- [Preparing the dataset](#Preparing-the-dataset)
- [Training the model](#Training-the-model)
- [Inference](#Inference)
- [Licence](#Licence)
- [Citation](#Citation)


## **Installation**
To set up the conda environment:

```bash
conda create -n dl python=3.10 -y
conda activate dl

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install e3nn glob2 tqdm h5py numpy scipy wandb scikit-learn biopython
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
## **Preparing the dataset**
1. Obtain the BBFlow ensemble, MD cluster representatives, and the FASTA file containing labels from Zenodo:
https://doi.org/10.5281/zenodo.18414050
2. Run `1_get_xyz_labels.py` to extract sequences and XYZ coordinates from the PDB files and combine them with the labels. This produces the primary HDF5 dataset.
3. Run `2_build_dataset_dynamic_augmentation_v2.py` to extract graph features and format the final HDF5 file used by the dataloader. Use the following parameters:
   - `--input`: Path to the primary `.h5` file (output of step 2)
   - `--output`: Path and name of the output `.h5` file
   - `--min-sequence-length`: Minimum sequence length to process (integer)
   - `--num-workers`: Number of worker processes for parallel data loading (integer)
```bash
python 2_build_dataset_dynamic_augmentation_v2.py \
       --input coords_labels.h5 \
       --output dataset.h5 \
       --num-workers 4
```

## **Training the model**

Follow these steps to configure and execute the two-stage model training pipeline:

1. Before running the training scripts, update the path to dataset and train/validation splits in `architecture/config.py`.
2. Train the primary geometric encoder model by specifying your desired model name:
   `python train.py <name_of_first_model>`
3. Run train_set.py to automatically extract latent embeddings using your pre-trained geometric encoder and train the set aggregation model:
   `python train_set.py <name_of_first_model> <name_of_second_model>`

## **Inference**

After setting up the Conda environment, inference can be performed using the `inference.py` script.

The pretrained models used for CryptoBench are:
- `model_4.pt` — geometric encoding  
- `model_44.pt` — set aggregation  

Alternatively, you can run inference directly using the Google Colab notebook:  
[CryptoBench Inference (Colab)](https://colab.research.google.com/drive/1FSkESlIdVksg2d0eBlj0E1AmxI9Hc1bc#scrollTo=Yt1l43M19bLq)

## **Licence**

Copyright (c) 2026 Omid Mokhtari, Inria

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## **Citation**
