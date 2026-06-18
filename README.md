# SIMORGH

![Ensemble Aggregation](Birds.gif)

## Table of Contents: 
- [The workflow](#The-workflow)
- [Installation](#Installation)
- [Training the model](#Training-the-model)
- [Inference](#Inference)
- [License](#License)
- [Acknowledgement](#Acknowledgement)
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
## **Training the model**
Follow these steps to train the model:
1. Access the clusters of MD and AFlow data from Zenodo:
https://doi.org/10.5281/zenodo.14833854
2. Extract sequences and XYZ coordinates from PDB files, and combine them with the labels to create an HDF5 dataset using get_xyz_labels.py.
3. Run the build_dataset.py script to generate an HDF5 (.h5) file. Use the following parameters:
   - ``--input``: path to the folder containing structures.
   - ``--output``: Path and name of the output .h5 file.
   - ``--min-sequence-length``:  Minimum sequence length to process (integer).
   - ``--num--workers``:  number of worker processes for data loading (integer).
```bash
python3 build_dataset.py --input /path/to/structures --output dataset.h5 --num-workers 4
```


## **Inference**

After setting up the Conda environment, inference can be performed using the `inference.py` script.

The pretrained models used for CryptoBench are:
- `model_4.pt` — geometric encoding  
- `model_44.pt` — set aggregation  

Alternatively, you can run inference directly using the Google Colab notebook:  
[CryptoBench Inference (Colab)](https://colab.research.google.com/drive/1FSkESlIdVksg2d0eBlj0E1AmxI9Hc1bc#scrollTo=7sNBpIvPzRgU)

## **License**

## **Citation**
