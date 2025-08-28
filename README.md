# BigEarthNet - Deep Learning Models

This repository is an implementation of DL models to multi-label classification of LULC classes present in the BigEarthNet Dataset. 

The repo contains the end-to-end AI pipeline for Data Pre Processing; Imbalancing; Dataset Classes; DataLoaders; Training, Validation and Test.

# Quick Start! 

## Download Dataset 

The download version matches the Version 2 (V2) of Big Earth, which corresponds to 19 classes of LULC and can be seen at [LULC CLASSES]("https://bigearth.net/static/documents/BigEarthNet_v2_Split.pdf")

## Create the enviroment 

Currently using uv on a Linux server, but feel free to use Conda


1. Create an virtual environment and activate it
```bash 
## Starts the .venv
uv init 

source .venv/bin/activate 

## Sync in order to retrieve all libraries present in the environment
uv sync 
```

2. Download the dataset and the complements.

Every file and link can be found at: [BigEarthNet]("https://bigearth.net/")

Caution for size: 70GB! 

```bash

mkdir bigearthnet-dataset

curl -L -o BigEarthNet-S2 --progress-bar link_dataset_BigEarthS2

curl -L -o metadata.parquet --progress-bar link-metadata-parquet

curl -L -o metadata_snow_cloud.parquet --progress-bar link_metadata_cloud_snow 
```

3. Test the download of the Dataset.

To test it, you only need to pass the path of the folder holding the dataset and the path of the parquet metadata.

Use the Notebook - [BigEarthNet-early-test](notebooks/bigearthnet-early-test.ipynb) to see if the dataset is correctly installed! 


-------------------------------------------------

## Folder Structure

```
BigEarthNet-Models/
┣ bigearthnet/
┃ ┗ explore/
┃   ┣ src/
┃ ┃ ┃ ┣ config/
┃ ┃ ┃ ┃ ┗ config.yaml
┃ ┃ ┃ ┣ list_config/ ## List of configs to run over a loop
┃ ┃ ┃ ┣ loader/
┃ ┃ ┃ ┃ ┗ reader.py  ## Dataset and DataLoader Class
┃ ┃ ┃ ┣ metrics/
┃ ┃ ┃ ┃ ┗ metrics.py ## Metrics
┃ ┃ ┃ ┣ model_zoo/
┃ ┃ ┃ ┃ ┣ classification.py
┃ ┃ ┃ ┃ ┗ models.py
┃ ┃ ┃ ┣ save_configs/ ## Other configs file
┃ ┃ ┃ ┗ utils/
┃ ┃ ┃   ┣ torch.py ## Function related to Pytorch
┃ ┃ ┃   ┣ utils.py  ## Utilities
┃ ┃ ┃   ┗ wandb_logger.py
┃   ┣ wandb/ ## Wandb logging files
┃   ┣ Explore2.ipynb ## Exploration of the dataset. 
┃   ┣ Explore_dataset.ipynb ## Explore dataset and a bit of imbalance
┃   ┣ define_model.ipynb  ## Explore architecture of model
┃   ┣ imbalanced_classes.ipynb ## Define imbalance and partial dataset more balanced
┃   ┣ main.py ## Main, the whole orchestrator of this repo
┃   ┣ plot_results.ipynb ## Plot final results
┃   ┣ resnet.ipynb ## Implement resnet from scratch
┃   ┗ vit_explore.ipynb ## Explore vit DL definitions
┣ df_balanced/ 
┃ ┗ teste2.parquet  ## Better balanced fragment of the whole dataset. Ideal for first models run
┣ notebooks/
┃ ┗ bigearthnet-early-test.ipynb  ## Test-suite
┣ plots/ ## Few plots
┣ README.md
┣ main.py
```


## Acknowledgements

This dataset is a personal project developed during my internship under [Destination Earth](https://platform.destine.eu/) initiative within [ESA](https://www.esa.int/).
A big thanks to [Sebastien Tetauld](https://github.com/sebastien-tetaud) for the inspiration and guidance. 
