from torch import nn
from torchgeo import models


def define_model_torchgeo(
    name,
    out_channels=20,
    in_channel=12):

    # Get the model class dynamically based on name
    try:
        # Get the model class from segmentation_models_pytorch
        ModelClass = getattr(models, name)


        # Create the model
        model = ModelClass(
            in_chans=in_channel,
            num_classes=out_channels,
            pretrained=True
        )

        ## Add softmax
        model = nn.Sequential(
                model,
                nn.Softmax(dim=1) ## over the class dimension 
            )
        return model


    except AttributeError:
        # If the model name is not found in the library
        raise ValueError(f"Model '{name}' not found in  torchgeo. Available models: {dir(models)}")
