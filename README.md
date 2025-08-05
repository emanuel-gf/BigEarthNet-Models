# BigEarthNet - Deep Learning Models

This repository is an implementation of DL models to classify LULC classes present on the BigEarthNet Dataset. 

## Download Dataset 

Please follow the instructions present at [download dataset]("install_dataset.md") to download it and config the BigEarthNet Dataset

The download version matches the Version 2 (V2) of Big Earth, which corresponds to 19 classes of LULC and can be seen at [LULC CLASSES]("https://bigearth.net/static/documents/BigEarthNet_v2_Split.pdf")

## Create the enviroment 

Currently using uv to manage the environments, but feel free to use it Conda
```bash 
## Starts the .venv
uv init 

## Sync in order to retrieve all libraries present in the environment
uv sync 
```