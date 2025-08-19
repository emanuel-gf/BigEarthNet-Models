from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Tuple
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union
import numpy as np
import pandas as pd
import tifffile
from loguru import logger
import lmdb
from safetensors.numpy import load as safetensor_load
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn.functional as F
from sklearn.preprocessing import minmax_scale

## Global  variables
means_s2 = {
    "120_nearest": {
        "B01": 361.0767822265625,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.8433227539062,
        "B06": 1769.931640625,
        "B07": 2049.551513671875,
        "B08": 2193.2919921875,
        "B09": 2241.455322265625,
        "B11": 1568.226806640625,
        "B12": 997.7324829101562,
        "B8A": 2235.556640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
    "120_bilinear": {
        "B01": 360.64678955078125,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.7476806640625,
        "B06": 1769.8486328125,
        "B07": 2049.475830078125,
        "B08": 2193.2919921875,
        "B09": 2241.10595703125,
        "B11": 1568.2115478515625,
        "B12": 997.715087890625,
        "B8A": 2235.48681640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    },
    "120_bicubic": {
        "B01": 360.637451171875,
        "B02": 438.3720703125,
        "B03": 614.0556640625,
        "B04": 588.4096069335938,
        "B05": 942.7472534179688,
        "B06": 1769.8485107421875,
        "B07": 2049.475830078125,
        "B08": 2193.2919921875,
        "B09": 2241.091064453125,
        "B11": 1568.2117919921875,
        "B12": 997.715087890625,
        "B8A": 2235.48681640625,
        "VH": -19.352558135986328,
        "VV": -12.643863677978516,
    }
}
stds_s2 = {
        "120_nearest": {
        "B01": 575.0687255859375,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.4326782226562,
        "B06": 1100.4560546875,
        "B07": 1275.805419921875,
        "B08": 1369.3717041015625,
        "B09": 1316.393310546875,
        "B11": 1070.1612548828125,
        "B12": 813.5276489257812,
        "B8A": 1356.5440673828125,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
    "120_bilinear": {
        "B01": 563.1734008789062,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 727.5784301757812,
        "B06": 1087.4288330078125,
        "B07": 1261.4302978515625,
        "B08": 1369.3717041015625,
        "B09": 1294.35546875,
        "B11": 1063.9197998046875,
        "B12": 806.8846435546875,
        "B8A": 1342.490478515625,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    },
    "120_bicubic": {
        "B01": 572.3436889648438,
        "B02": 607.02685546875,
        "B03": 603.2968139648438,
        "B04": 684.56884765625,
        "B05": 738.3037719726562,
        "B06": 1100.46142578125,
        "B07": 1275.843505859375,
        "B08": 1369.3717041015625,
        "B09": 1313.6488037109375,
        "B11": 1070.8011474609375,
        "B12": 814.0936279296875,
        "B8A": 1356.754150390625,
        "VH": 5.590505599975586,
        "VV": 5.133493900299072,
    }
}
_s2_bandnames_10m_20m = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
_s2_bandnames = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]



def get_right_dict(s2_dict_mean,s2_dict_std, upsampling_method='nearest'):       
        name_upsample = f"120_{upsampling_method}"
        mean_dict = s2_dict_mean[name_upsample]
        std_dict = s2_dict_std[name_upsample]
        return mean_dict,std_dict

def get_list_means_std(mean_dict,std_dict,strip_bands):
        """Extract mean and std values for each band from dictionaries.
        args:
            strip_bands: list[str]
                Expect a band with the name of the bands inside e.g [B02,B03]
"""
        list_mean, list_std = [],[]
        for band in strip_bands:
            list_mean.append(mean_dict[band])
            list_std.append(std_dict[band])
            
        return list_mean, list_std









class Reader:
    
    def __init__(self, root_folder_path: str, metadata_parquet_path: str):
        """     
    Read all tif files inside subfolder structure from a given root folder. It returns labels and bands for each patch.
        Args:
            root_folder_path (str): Path to the root folder to analyze
            metadata_parquet_path (str): Path to the metadata parquet file

        """
        logger.info('Initializing the Reader, structuring the paths and files.')
        self.root_folder_path = Path(root_folder_path).expanduser().resolve()
        self.metadata_parquet = pd.read_parquet(metadata_parquet_path)


        
        # Validate the root folder exists and is a directory
        if not self.root_folder_path.exists():
            raise FileNotFoundError(f"Root folder does not exist: {self.root_folder_path}")
        if not self.root_folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root_folder_path}")
        
        # Initialize folder lists
        self._parent_folders: Optional[List[str]] = None
        self._all_subfolders: Optional[List[List[str]]] = None
        
        # Initialize plain_full_sublist automatically
        self.plain_full_sublist = [item for sublist in self.all_subfolders for item in sublist]
        
        # Create optimized lookups for faster access
        self._create_lookup_structures()

        logger.info(self.get_total_folder_count())

    def _create_lookup_structures(self):
        """Create optimized lookup structures for faster access."""
        # Create patch_id to labels lookup for O(1) access
        self._patch_labels_dict = dict(zip(self.metadata_parquet['patch_id'], self.metadata_parquet['labels']))
        
        # Create patch_id to parent folder mapping for faster path resolution
        self._patch_to_parent = {}
        for parent_folder in self.parent_folders:
            subfolder_path = self.root_folder_path / parent_folder
            if subfolder_path.exists():
                for subfolder in subfolder_path.iterdir():
                    if subfolder.is_dir():
                        self._patch_to_parent[subfolder.name] = parent_folder
        
        logger.info(f"Created lookup structures for {len(self._patch_labels_dict)} patches")

    @property
    def parent_folders(self) -> List[str]:
        """
        Get all directory names inside the root folder.
        
        Returns:
            List[str]: List of directory names in the root folder
        """
        if self._parent_folders is None:
            try:
                self._parent_folders = [
                    f.name for f in self.root_folder_path.iterdir() 
                    if f.is_dir()
                ]
                logger.info(f"Found {len(self._parent_folders)} folders in {self.root_folder_path}")
            except PermissionError:
                logger.error(f"Permission denied accessing {self.root_folder_path}")
                self._parent_folders = []
            except Exception as e:
                logger.error(f"Error reading parent folders: {e}")
                self._parent_folders = []
        
        return self._parent_folders
    
    @property
    def all_subfolders(self) -> List[List[str]]:
        """
        Get all subdirectories inside each parent folder.
        
        Returns:
            List[List[str]]: List where each element is a list of subfolder names
                           for each parent folder
        """
        if self._all_subfolders is None:
            self._all_subfolders = []
            
            for parent_folder in self.parent_folders:
                subfolder_path = self.root_folder_path / parent_folder
                try:
                    subfolders = [
                        f.name for f in subfolder_path.iterdir() 
                        if f.is_dir()
                    ]
                    self._all_subfolders.append(subfolders)
                    #logger.debug(f"Found {len(subfolders)} subfolders in {subfolder_path}")
                except PermissionError:
                    logger.warning(f"Permission denied accessing {subfolder_path}")
                    self._all_subfolders.append([])
                except Exception as e:
                    logger.warning(f"Error reading subfolders in {subfolder_path}: {e}")
                    self._all_subfolders.append([])
        
        return self._all_subfolders
    
    def get_folder_structure(self) -> dict:
        """
        Get folder structure as a nested dictionary.
        
        Returns:
            dict: Nested dictionary representing the folder structure
        """
        structure = {}
        
        for i, parent_folder in enumerate(self.parent_folders):
            if i < len(self.all_subfolders):
                structure[parent_folder] = self.all_subfolders[i]
            else:
                structure[parent_folder] = []
        
        return structure
    
    def get_total_folder_count(self) -> int:
        """
        Get the total number of folders (parent + all subfolders).
        
        Returns:
            int: Total number of folders found
        """
        parent_count = len(self.parent_folders)
        subfolder_count = len(self.plain_full_sublist)
        print(f"Number of parent folders: {parent_count}")
        print(f"Number of total patches found it: {subfolder_count}")

    
    def load_tif_bands(self, folder_path: Path) -> Dict[str, any]:
        """
        The folder path is a the same string presented at the plain_full_sublist
        Loads all .tif files in a folder and returns a dict with band names as keys and numpy arrays as values.
        Assumes band name is part of the filename, e.g., 'patch_B01.tif' -> band 'B01'.
        """
        band_dict = {}
        for tif_file in folder_path.glob("*.tif"):
            # Extract band name from filename (e.g., 'B01' from 'patch_B01.tif')
            band = [part for part in tif_file.stem.split('_') if part.startswith('B')]
            if band:
                band_name = band[0]
                band_dict[band_name] = tifffile.imread(str(tif_file))
        return band_dict
    
    
    def __getitem__(self, patch_id: str) -> Tuple[Dict[str, any], any]:
        """
        Returns a tuple containing band data and labels for a given patch ID.
        
        Args:
            patch_id (str): The patch ID (e.g., 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57')
                           which should match a folder name in plain_full_sublist
        
        Returns:
            Tuple[Dict[str, any], any]: Tuple of (bands_dict, labels)
        
        Raises:
            ValueError: If patch_id is not found in the folder structure or metadata
        """
        # Fast O(1) lookup for labels
        if patch_id not in self._patch_labels_dict:
            raise ValueError(f"Patch ID '{patch_id}' not found in metadata parquet")
        
        labels = self._patch_labels_dict[patch_id]
        
        # Fast O(1) lookup for parent folder
        if patch_id not in self._patch_to_parent:
            raise ValueError(f"Patch ID '{patch_id}' not found in folder structure")
        
        parent_folder = self._patch_to_parent[patch_id]
        subfolder_path = self.root_folder_path / parent_folder / patch_id
        
        if not subfolder_path.exists():
            raise ValueError(f"Patch ID '{patch_id}' folder not found at {subfolder_path}")
        
        bands = self.load_tif_bands(subfolder_path)
        return (bands, labels)
    
    def __str__(self) -> str:
        """String representation of the Reader instance."""
        return f"Reader(root='{self.root_folder_path}', parent_folders={len(self.parent_folders)})"
    
    def __repr__(self) -> str:
        """Developer representation of the Reader instance."""
        return f"Reader(root_folder_path='{self.root_folder_path}')"
    


class Dataset_BigEarthNet:
    def __init__(
        self,
        reader,   # class
        strip_bands,
        img_size,
        upsample_mode,
        normalize,
        split_train_test,
        small_fraction = None,
        transform=None,
        return_patch_id=False,
        patch_ids=None,
        dict_one_hot={},

    ):
        """
        Implements a Dataset submodule of pytorch. 

        Args:
            reader: Reader CLASS
                See Reader class
            patch_ids: 
                The patch_ids. The way of selecting Train, Test and Val patches. 
            strip_bands: List. 
                Bands to be selected
            img_size: int.
                Size of the Image. It will apply a interpolation to the desired image size.
            upsample_mode: str Nearest, Bicubic, Bilinear
                Method of interpolation
            band_order: List 
                Order of the bands presented at the BigEarthDataset. 
            normalize: bool
                If normalize the data or not.
            transform:
                If apply torch.transform on images
            return_patch_id: bool
                To return the patch Index or not
            split_train_test: str
                Split the idx list into the choice, either: Train, Test or Val
            dict_one_hot: dict
                Dict of labels for one-hot encoding
            
        """
        
        self.reader = reader
        self.patch_ids = patch_ids
        self.strip_bands = strip_bands
        self.img_size = img_size
        self.upsample_mode = upsample_mode
        self.normalize = normalize
        self.split_train_test = split_train_test
        self.transform = transform
        self.return_patch_id = return_patch_id
        self.small_fraction = small_fraction

        ## This store the MultiLabelBinarizer, which is the class to transform in one-hot-encoding
        self.mlb_labels_  = None
        self.mlb = MultiLabelBinarizer().fit(self.reader.metadata_parquet['labels'].values)

        self.dict_one_hot = {v:i for i,v in enumerate(self.mlb.classes_)}
        self.inverse_dict_one_hot = {i:v for i,v in enumerate(self.mlb.classes_)}

        # Initialize patch_ids_array based on split_train_test
        if self.split_train_test is not None:
            # Filter the df to train/test/val split
            self.patch_ids_array = self.__get_patch_id_array__()
            logger.info(f"{self.split_train_test} split, with length of {len(self.patch_ids_array)}")
        else:
            # Use all patch_ids from metadata_parquet
            self.patch_ids_array = self.reader.metadata_parquet['patch_id'].values
            logger.info(f"Full dataset, with length of {len(self.patch_ids_array)}")

        if self.small_fraction is not None:
            self.patch_ids_array = self.__fraction_dataset__()
            logger.warning(f"Small sample selected. Dataset is reduced!")
            logger.info(f"{self.split_train_test} split, with length of {len(self.patch_ids_array)}")
        else:
            logger.warning("Using full dataset!!")

        logger.info(f"Bands orded by: {strip_bands}")

    def __len__(self) -> int:
        return len(self.patch_ids_array)
    
    def __patch_id__(self, index: int) -> str:
        return self.patch_ids_array[index]
    
    def __series_patch_id__(self) -> pd.Series:
        return pd.Series(self.patch_ids_array)
    
    def __get_patch_id_array__(self):
        """Filter the metadata to get patch IDs for the specified split (train,test,validation)"""
        if hasattr(self.reader, 'metadata_parquet') and 'split' in self.reader.metadata_parquet.columns:
            array = self.reader.metadata_parquet.loc[
                                                    self.reader.metadata_parquet['split'] == self.split_train_test
                                                    ]['patch_id'].values
            return array
        else:
            logger.error("Reader metadata doesn't contain 'split' column or metadata is missing")
            return []
    
    def __fraction_dataset__(self):
        size_fraction = int(len(self.patch_ids_array)*self.small_fraction)
        old= self.patch_ids_array
        new_patch_ids_array = old[:size_fraction]
        return new_patch_ids_array

    def calculate_unbalanced_df(self):
        """
        Calculates the tensor of positional weight to be passed at the BCELogitLoss() function.
        This tensor is given by the ratio of neg/pos classes. And deals with imbalanced classes.
        The idea is that less frequent classes have a higher weight applied to the loss calculation.
        However, the upper limit of the clip output should be carefully adapt it
        """
        ## Get the filtered df by train/split and by the fraction applyied
        filtered_df = self.reader.metadata_parquet.loc[self.reader.metadata_parquet['patch_id'].isin(self.patch_ids_array)]

        labels_matrix = self.mlb.transform(filtered_df['labels'])
        
        ## Convert to df
        labels_df = pd.DataFrame(labels_matrix, columns=self.mlb.classes_)

        ## Sum classes
        df_sum = pd.DataFrame()
        df_sum['sum'] = labels_df.sum(0).values
        ##df_sum.sort_values(by='labels',inplace=True)  ##MAKE IT SURE IS IN ORDER.
        df_sum['num_negatives'] = (labels_df.T.sum(1).sum()- labels_df.T.sum(1)).values
        df_sum['neg/pos'] = df_sum['num_negatives']/df_sum['sum']

        ## apply square root of log to use as the weights
        #df_sum["logs"] = np.sqrt(np.log(df_sum["neg/pos"].values + 1)) 
        df_sum["logs"] = np.log(df_sum["neg/pos"].values + 1) 
        logger.info(f"Value of imbalanced before clip: {df_sum['logs'].values}")
        ## scale over 1-10 
        array_out = minmax_scale(df_sum["logs"].values, feature_range=(1,8)).astype(int)

        print('---'*15)
        if self.inverse_dict_one_hot is not None:
            for i,v in enumerate(array_out):
                print(self.inverse_dict_one_hot[i],':',v)

        logger.warning(f"Unbalanced results: {array_out}")
        return torch.tensor(array_out)
    
    def return_mlb_classes(self):
        """ Return a MultiLabelBinarizer Class
        """
        arr = self.mlb.classes_
        return arr
    
    def return_mlb(self):
        return self.mlb
    
    def __length__(self):
        return len(self.patch_ids_array)
    
    def stack_and_interpolate(
        self,
        bands: Dict[str, np.ndarray],
        order: Optional[Iterable[str]] = None,
        img_size: Optional[int] = None,
        upsample_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Stack and interpolate bands according to the specified order.
        
        Args:
            bands: Dictionary of band data where keys are band names and values are numpy arrays
            order: Order of bands to stack. If None, uses self.strip_bands
            img_size: Target image size. If None, uses self.img_size
            upsample_mode: Interpolation mode. If None, uses self.upsample_mode
            
        Returns:
            torch.Tensor: Stacked and interpolated bands
        """
        # Use instance variables as defaults
        if order is None:
            order = self.strip_bands
            ##logger.info(f"Using order of bands: {order}")
        if img_size is None:
            img_size = self.img_size
        if upsample_mode is None:
            upsample_mode = self.upsample_mode


        def _interpolate(img_data):
            """Helper function to interpolate individual band data"""
            if not img_data.shape[-2:] == (img_size, img_size):
                return F.interpolate(
                    torch.Tensor(np.float32(img_data)).unsqueeze(0).unsqueeze(0),
                    (img_size, img_size),
                    mode=upsample_mode,
                    align_corners=True if upsample_mode in ["nearest","bilinear", "bicubic"] else None,
                ).squeeze()
            else:
                return torch.Tensor(np.float32(img_data))
        
        # Order and strip bands - only include bands that exist in the data
        available_bands = []
        for band in order:
            if band in bands.keys():
                available_bands.append(band)
            else:
                logger.warning(f"Band {band} not found in available bands: {list(bands.keys())}")
        
        if not available_bands:
            raise ValueError(f"No bands from the specified order {order} were found in the data")
        
        # Stack bands in the correct order
        stacked_bands = torch.stack([_interpolate(bands[band]) for band in available_bands])
        
        return stacked_bands

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, str]]:
        """Get item by index"""
        # Get patch_id from the array
        patch_id = self.patch_ids_array[idx]

        try:
            # Get data from LMDB reader
            image_data, labels = self.reader.__getitem__(patch_id)

            # Convert labels to one-hot encoder if available
            if self.mlb is not None:
                labels_onehot = self.mlb.transform([labels])
                labels_onehot = torch.from_numpy(labels_onehot).squeeze(0).float()
            else:
                # If no MLBinarizer, return labels as-is
                labels_onehot = labels

            # Stack and Interpolate using the new method
            img_tensor = self.stack_and_interpolate(bands=image_data)

            # Apply transforms if provided
            if self.transform is not None:
                img_tensor = self.transform(img_tensor)

            # Return data
            if self.return_patch_id:
                return img_tensor, labels_onehot, patch_id
            else:
                return img_tensor, labels_onehot
                
        except Exception as e:
            logger.error(f"Error loading patch {patch_id}: {str(e)}")
            raise
