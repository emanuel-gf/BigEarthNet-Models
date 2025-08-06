import segmentation_models_pytorch as smp
import torch.nn as nn
import timm
import torch
from typing import Union, Optional
from torchgeo.models import get_weight
from torchgeo.models.api import WeightsEnum
from torch.hub import load_state_dict_from_url
import os 
import re 
from loguru import logger


def define_model(
    name,
    encoder_name,
    out_channels=3,
    in_channel=3,
    encoder_weights=None,
    activation=None,

):
    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(smp, name)


        # Create the model
        model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channel,
            classes=out_channels,
            activation=None,

        )

        # Add ReLU activation after the model
        if activation == "relu":
            model = nn.Sequential(
                model,
                nn.ReLU()
            )
        if activation == "sigmoid":
            model = nn.Sequential(
                model,
                nn.Sigmoid()
            )



        return model


    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in segmentation_models_pytorch. Available models: {dir(smp)}")


def load_state_dict_with_flexibility(model: nn.Module, state_dict: dict, strict: bool = False, 
                                    bands: dict = None, selected_channels: list=None):
    """
    Load state dict with flexible key matching and size adaptation.
    
    Args:
        model: Target model to load weights into
        state_dict: State dictionary to load
        strict: Whether to enforce strict key matching
        bands: Optional mapping of channel indices to satellite band names
                        e.g., {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", ...}
        selected_channels: List of selected channels to stay over the mapping 
    """
    model_state_dict = model.state_dict()
    
    # Create mapping between state_dict keys and model keys
    key_mapping = {}
    unmatched_keys = []
    
    for state_key in state_dict.keys():
        # Try exact match first
        if state_key in model_state_dict:
            key_mapping[state_key] = state_key
            continue
        
        # Try to find similar keys (handle different naming conventions)
        matched = False
        for model_key in model_state_dict.keys():
            if _keys_match(state_key, model_key):
                key_mapping[state_key] = model_key
                matched = True
                break
        
        if not matched:
            unmatched_keys.append(state_key)
    
    logger.info('Adapating Tensor Input...')
    # Filter and adapt state dict
    adapted_state_dict = {}
    for state_key, model_key in key_mapping.items():
        state_tensor = state_dict[state_key]
        model_tensor = model_state_dict[model_key]
        
        # Handle size mismatches
        if state_tensor.shape != model_tensor.shape:
            adapted_tensor = _adapt_tensor_size(state_tensor, model_tensor, state_key, bands, selected_channels)
            if adapted_tensor is not None:
                adapted_state_dict[model_key] = adapted_tensor
                logger.info(f" Adapted {state_key} -> {model_key}: {state_tensor.shape} -> {adapted_tensor.shape}")
            else:
                print(f" Skipping {state_key} -> {model_key}: incompatible shapes {state_tensor.shape} vs {model_tensor.shape}")
        else:
            adapted_state_dict[model_key] = state_tensor
    
    # Load the adapted state dict
    missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=strict)
    
    print(f"\n📊 LOADING SUMMARY:")
    print(f"✅ Successfully loaded: {len(adapted_state_dict)} parameters")
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)} (these will use random initialization)")
    if unexpected_keys:
        print(f"🔄 Unexpected keys: {len(unexpected_keys)} (these were ignored)")
    if unmatched_keys:
        print(f"❓ Unmatched keys from source: {len(unmatched_keys)}")
    
    return missing_keys, unexpected_keys


def _keys_match(state_key: str, model_key: str) -> bool:
    """Check if two keys represent the same parameter with different naming conventions."""
    # Remove common prefixes/suffixes that might differ
    state_clean = re.sub(r'^(backbone\.|encoder\.|features\.)', '', state_key)
    model_clean = re.sub(r'^(backbone\.|encoder\.|features\.)', '', model_key)
    
    # Check for exact match after cleaning
    if state_clean == model_clean:
        return True
    
    # Check for common substitutions
    substitutions = [
        (r'\.weight$', '.weight'),
        (r'\.bias$', '.bias'),
        (r'bn(\d+)', r'norm\1'),  # batch norm naming
        (r'norm(\d+)', r'bn\1'),
        (r'downsample\.0', 'downsample.conv'),
        (r'downsample\.1', 'downsample.norm'),
    ]
    
    for pattern, replacement in substitutions:
        if re.sub(pattern, replacement, state_clean) == model_clean:
            return True
        if state_clean == re.sub(pattern, replacement, model_clean):
            return True
    
    return False


def _adapt_tensor_size(state_tensor: torch.Tensor, target_tensor: torch.Tensor, key_name: str, 
                      channel_mapping: dict = None, selected_channels: list = None) -> Union[torch.Tensor, None]:
    """
    Adapt tensor size to match target, handling common mismatches with specific channel selection.
    
    Args:
        state_tensor: Source tensor from state dict
        target_tensor: Target tensor from model
        key_name: Name of the parameter for context
        channel_mapping: Optional mapping of channel indices to band names
        selected_channels: Optional list of specific channel indices to keep/select
                          e.g., [0, 1, 2, 3, 5, 7] to skip channels 4 and 6
                          If None, defaults to first N channels
    
    Returns:
        Adapted tensor or None if incompatible
    """
    state_shape = state_tensor.shape
    target_shape = target_tensor.shape
    
    # Handle input channel adaptation (common for first conv layer)
    if 'conv' in key_name.lower() and len(state_shape) == 4 and len(target_shape) == 4:
        if state_shape[1] != target_shape[1]:  # Different input channels
            print(f"\SATELLITE BAND ADAPTATION for {key_name}")
            print(f"Source channels: {state_shape[1]} -> Target channels: {target_shape[1]}")
            
            # Create default channel mapping if not provided
            if channel_mapping is None:
                # Common satellite band mappings (adjust based on your satellite data)
                common_bands = {
                    0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", 5: "SWIR2",
                    6: "RedEdge1", 7: "RedEdge2", 8: "RedEdge3", 9: "NIR2", 
                    10: "Coastal", 11: "Cirrus", 12: "TIR1", 13: "TIR2"
                }
                channel_mapping = {i: common_bands.get(i, f"Band_{i}") for i in range(state_shape[1])}
            
            logger.info(f"Available bands in source weights:",level='INFO')
            for i in range(state_shape[1]):
                band_name = channel_mapping.get(i, f"Band_{i}")
                logger.info(f"  Channel {i:2d}: {band_name}",level='INFO')
            
            if target_shape[1] < state_shape[1]:
                # Need to remove channels - use specific selection if provided
                if selected_channels is not None:
                    # Validate selected channels
                    if len(selected_channels) != target_shape[1]:
                        logger.error(f"selected_channels length ({len(selected_channels)}) must match target_shape[1] ({target_shape[1]})")
                        raise ValueError(f"selected_channels length ({len(selected_channels)}) must match target_shape[1] ({target_shape[1]})")
                         
                    if max(selected_channels) >= state_shape[1] or min(selected_channels) < 0:
                        logger.error(f"selected_channels indices must be in range [0, {state_shape[1]-1}]")
                        raise ValueError(f"selected_channels indices must be in range [0, {state_shape[1]-1}]")
                    
                    # Remove duplicates and sort for display
                    unique_selected = list(set(selected_channels))
                    if len(unique_selected) != len(selected_channels):
                        raise ValueError("selected_channels contains duplicate indices")
                    
                    # Show what's being selected and removed
                    all_channels = set(range(state_shape[1]))
                    removed_channels = sorted(all_channels - set(selected_channels))
                    
                    print(f"REMOVING {len(removed_channels)} specific channels:")
                    logger.info(f"REMOVING {len(removed_channels)} specific channels:")

                    for ch in removed_channels:
                        band_name = channel_mapping.get(ch, f"Band_{ch}")
                        print(f" Removing Channel {ch:2d}: {band_name}")
                    
                    print(f"SELECTING {len(selected_channels)} channels (in order):")
                    for i, ch in enumerate(selected_channels):
                        band_name = channel_mapping.get(ch, f"Band_{ch}")
                        print(f"  Position {i:2d} <- Channel {ch:2d}: {band_name}")
                    
                    # Select specific channels using advanced indexing
                    # This preserves the order specified in selected_channels
                    selected_tensor = state_tensor[:, selected_channels, :, :]
                    
                    logger.success(f"Sucessfully selected Bands. Selected order: {selected_channels}")
                    print(f"CUSTOM CHANNEL SELECTION APPLIED")
                    print(f"   Original order: {list(range(state_shape[1]))}")
                    print(f"   Selected order: {selected_channels}")
                    
                
                else:
                    print("Please provide a list of channels to be selected.")
                
                return selected_tensor
  
            
    # Handle classifier/head layer adaptation
    if any(classifier_name in key_name.lower() for classifier_name in ['classifier', 'head', 'fc']):
        if len(state_shape) == 2 and len(target_shape) == 2:
            if state_shape[0] != target_shape[0]:  # Different number of classes
                logger.error(f"Skipping classifier layer {key_name} due to class mismatch: {state_shape[0]} vs {target_shape[0]}")
                print(f"Skipping classifier layer {key_name} due to class mismatch: {state_shape[0]} vs {target_shape[0]}")
                return None
    
    # Handle 1D tensors (bias, batch norm parameters)
    if len(state_shape) == 1 and len(target_shape) == 1:
        if state_shape[0] != target_shape[0]:
            logger.error(f"Skipping 1D parameter {key_name} due to size mismatch: {state_shape[0]} vs {target_shape[0]}")
            print(f"Skipping 1D parameter {key_name} due to size mismatch: {state_shape[0]} vs {target_shape[0]}")
            return None
    
    return None


def define_model_(
        model_name: str,
        num_classes: int,
        input_channels: int = 3,
        freeze_backbone: bool = False,
        weights: Union[str, WeightsEnum, None, bool] = None,
        bands: dict = None,
        selected_channels = list,
        **kwargs) -> nn.Module:
    """
    Create a model with PyTorchGeo weights support.
    
    Args:
        model_name: Name of the model architecture (e.g., 'resnet50', 'efficientnet_b0')
        weights: Weight loading strategy:
            - True: Load ImageNet pretrained weights via TIMM
            - WeightsEnum: Load specific PyTorchGeo weights
            - str: Either a path to weights file or PyTorchGeo weight name
            - None/False: No pretrained weights
        num_classes: Number of output classes
        input_channels: Number of input channels (default: 3 for RGB)
        freeze_backbone: Whether to freeze backbone parameters
        bands: Optional mapping of channel indices to satellite band names
                        e.g., {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", ...}
                        Common satellite sensors:
                        - Sentinel-2: {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", 5: "SWIR2", ...}
                        - Landsat-8: {0: "Coastal", 1: "Blue", 2: "Green", 3: "Red", 4: "NIR", 5: "SWIR1", 6: "SWIR2", ...}
        **kwargs: Additional arguments passed to timm.create_model
    
    Returns:
        torch.nn.Module: Configured model with loaded weights
    """
    use_timm_pretrained_weights_imagenet = weights is True
    
    logger.info(f'Creating Model: {model_name} with weights: {weights}')

    # Create model with TIMM
    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        in_chans=input_channels,
        pretrained=use_timm_pretrained_weights_imagenet,
        **kwargs
    )
    
    logger.info("Loading weights")
    # Load PyTorchGeo weights
    if weights and weights is not True:
        try:
            # Handle different weight types
            if isinstance(weights, WeightsEnum):
                logger.info(f"Loading PyTorchGeo weights: {weights}")
                state_dict = weights.get_state_dict(progress=True)

            elif isinstance(weights, str):
                if weights.endswith('.pth') or weights.endswith('.pt'):
                    # Load from file path
                    logger.info(f"Loading weights from file: {weights}")
                    state_dict = torch.load(weights, map_location='cpu')
                    # Handle different state dict formats
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                else:
                    # Load by PyTorchGeo weight name
                    logger.info(f"Loading PyTorchGeo weights by name: {weights}")
                    weight_enum = get_weight(weights)
                    state_dict = weight_enum.get_state_dict(progress=True)
            else:
                raise ValueError(f"Unsupported weight type: {type(weights)}")
            
            # Use flexible loading with satellite band information
            load_state_dict_with_flexibility(model, state_dict, strict=False, bands=bands,selected_channels=selected_channels)
            logger.success("✓ Weights loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            print("Continuing with model initialization...")
    
    # Handle backbone freezing
    if freeze_backbone:
        logger.info("Freezing backbone parameters...")
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier/head parameters
        try:
            # Try different common classifier attribute names
            classifier_attrs = ['classifier', 'head', 'fc']
            classifier = None
            
            for attr in classifier_attrs:
                if hasattr(model, attr):
                    classifier = getattr(model, attr)
                    break
            
            # Also try get_classifier method
            if classifier is None and hasattr(model, 'get_classifier'):
                try:
                    classifier = model.get_classifier()
                except:
                    pass
            
            if classifier is not None:
                for param in classifier.parameters():
                    param.requires_grad = True
                logger.info("✓ Classifier head unfrozen")
            else:
                logger.info("⚠ Warning: Could not find classifier to unfreeze")
                # Print available attributes for debugging
                logger.info(f"Available model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
                
        except Exception as e:
            logger.error(f"⚠ Warning: Could not unfreeze classifier - {e}")
    
    logger.info("Add softmax")
    ### Add the softmax 
    model = nn.Sequential(
                model,
                nn.Softmax(dim=1)
            )
    logger.success("Model is ready")
    return model


