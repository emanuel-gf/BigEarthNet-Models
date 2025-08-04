
## How to Download and Access the BigEarthNet Dataset

### Download and preparing the Dataset

The recomendations here are for using UV management on a Linux server. 

1. Create an virtual environment and activate it
```bash
uv venv 

source .venv/bin/activate

## Sync the libraries 
uv sync
```

2. Download the dataset and the complements.
Every file and link can be found at: [BigEarthNet]("https://bigearth.net/")

```bash
curl -L -o BigEarthNet-S2 --progress-bar link_dataset_BigEarthS2

curl -L -o reference_maps.tar.zst --progress-bar link_reference_map

curl -L -o metadata.parquet --progress-bar link-metadata-parquet

curl -L -o metadata_snow_cloud.parquet --progress-bar link_metadata_cloud_snow 
```

3.5 Decompress the necessary file
```bash 
tar --use-compress-program=unzstd -xvf Reference_Maps.tar.zst

```
3. Check if the libraries meet all the requirements

```bash
uv pip install configilm[bigearthnet]
uv pip install configilm[full-pytorch-lightning]
```

4. Convert the files to an Lighthining Database format. It uses an AppImage and the encoder [rico-hbl]("https://github.com/rsim-tu-berlin/rico-hdl?tab=readme-ov-file")

Download the AppImage from the github repository of rico-hbl. Check it this [link]("https://git.tu-berlin.de/rsim/reben-training-scripts/-/blob/main/README.md?ref_type=heads") in case you are on a Windows environment 
```bash 
curl -L -o AppImage "https://github.com/kai-tub/rico-hdl/releases/latest/download/rico-hdl.AppImage"
```

5. Access the app image to use it as a command-line

Change to the directory containing the AppImage 
```bash 
cd same-directory-as-the-AppImage

## Make the AppImage executable
chmod +x AppImage 

```

6. Run the Decoder

```bash
## create a target directory
mkdir encoder_lmdb

## Apply the rico-dbl AppImage
./AppImage bigearthnet --bigearthnet-s2-dir "BigEarthNet-S2" --bigearthnet-reference-maps-dir "Reference_Maps" --target-dir "encoder_lmdb"
```

7. Run on python

After all the steps above, the dataset is ready to be run. 
Go the getting started notebook to reach that 

8. To load and see the dataset do as the following:
```ipynb
## Create the config file
path_dataset = {
    "images_lmdb": "../encoder_lmdb",
    "metadata_parquet": "../metadata.parquet",
    "metadata_snow_cloud_parquet": "../metadata_snow_cloud.parquet"
}

## Load it as a class
ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=path_dataset
)

## Verify first image
img, lbl = ds[0]

```