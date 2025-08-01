import yaml
from urllib.parse import urlparse
#import pandas as pd
import os

def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
