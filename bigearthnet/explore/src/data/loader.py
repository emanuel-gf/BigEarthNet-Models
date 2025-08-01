from configilm import util
util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages

from configilm.extra.DataSets import BENv2_DataSet
from configilm.extra.DataModules import BENv2_DataModule

""" This data loader is specifically focused on the BigEarthNet dataset. 

    In order to reuse code from TU-Berlin and already implemented loader present at the library configILM,
     the classes provides a way to load the dataset.

     It is necessary the  ldmb folder, which the instructions are given on the readme file.  
"""


def bigearthnet_loader(
                        path_dataset_lmdb,
                        path_metadata_parquet,
                        path_metadata_snow_cloud_parquet,
                        train: bool,
                        max_img_idx = None
                        ):

    """
    Create a pytorch.DataLoader class. Wrapped over the configILM library. 

    Parameters
    --------------------------
    path_dataset_lmdb: str 
        Path to the lmdb folder. This folder is generated with the AppImage. Please read the readme for further orientation.
    path_metadata_parquet: str
         Path to the metadata .parquet
    path_metatada_snow_cloud: str
        Path to metadata .parquet
    train: bool
        Wheater TRAIN split or not the dataset. If true, applies the method present in the class BENv2_DataSet. This method implies the same
        split seed by default. 
    max_img_idx : int 
        Restrict the number of images retrieved. Only the first n images (in alphabetical order based on S2-name) will be loaded. 
    """
    ## Create the config dict 
    path_dataset = {
         "images_lmdb":path_dataset_lmdb,
        "metadata_parquet": path_metadata_parquet,
        "metadata_snow_cloud_parquet": path_metadata_snow_cloud_parquet
    }

    if (train) & (max_img_idx!=None):
        ds = BENv2_DataSet.BENv2DataSet(
            data_dirs=path_dataset,
            split="train"
        )
        return ds

    if (train==True) & (max_img_idx!=None):
        ds = BENv2_DataSet.BENv2DataSet(
        data_dirs=path_dataset,
        split="train",
        max_img_idx = max_img_idx
        )
        return ds

    else:
        ds = BENv2_DataSet.BENv2DataSet(
            data_dirs=path_dataset
        )

        return ds




def bigearthnet_DataModule(
                        path_dataset_lmdb,
                        path_metadata_parquet,
                        path_metadata_snow_cloud_parquet,
                        **kwargs
    ): 
    """
        This function implements the DataModule from configILM.  The DataModule is a wrapper of LighthingPytorch and automatically generates 
        per split with augmentations, shuffiling and other resources. All images are resized by default and normalized. The train set are additionally 
        augmented via noise and flipping/rotation. The train split is shuffled.  

        It accepts any kwargs present at the BENv2_DataModule. e.g(shuffle, nurm_workers_dataloader,batch_size, img_sizew)
    """ 

    my_data_path = {
         "images_lmdb":path_dataset_lmdb,
        "metadata_parquet": path_metadata_parquet,
        "metadata_snow_cloud_parquet": path_metadata_snow_cloud_parquet
    }
    dm = BENv2_DataModule.BENv2DataModule(
        data_dirs = my_data_path,
        **kwargs
    )
    return dm