import yaml
import os

from src.utils.utils import load_config
from src.data.loader import bigearthnet_loader, bigearthnet_DataModule

## Result dictionary 
def create_result_dirs(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(base_dir, timestamp)
    checkpoint_path = os.path.join(result_dir, "checkpoints")
    metrics_path = os.path.join(result_dir, "metrics")
    log_path = os.path.join(result_dir, "training.log")

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return {
        "timestamp": timestamp,
        "result_dir": result_dir,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "log_path": log_path
    }

## 
def setup_environment(config, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="10 MB")
    seed_everything(seed=config['TRAINING']['seed'])


def save_config_to_log(config, log_dir, filename="config.yaml"):
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, filename)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info(f"Saved config to {config_path}")


def prepare_paths(path_dir):


    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE","")))
    df_output["path"] = df_output["Name"].apply(lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE","")))

    return df_input, df_output


def build_model(config):


    model = define_model(
        name=config['MODEL']['model_name'],
        encoder_name=config['MODEL']['encoder_name'],
        encoder_weights = config['MODEL']['encoder_weights'],
        in_channel=len(config['DATASET']['bands']),
        out_channels=len(config['DATASET']['bands']),
        activation=config['MODEL']['activation'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, device


def main()->None:

    paths = create_result_dirs()
    log_path = paths['log_path']
    checkpoint_path = paths['checkpoint_path']
    metrics_path = paths['metrics_path']
    bands = config['DATASET']['bands']
    num_epochs = config['TRAINING']['n_epoch']
    # Initialize best metrics at the beginning of training
    if config['TRAINING']['save_strategy'] == "loss":
        best_metric = float('inf')  # For loss, lower is better
        logger.info("Model will be saved based on validation loss")
    else:  # metric-based saving
        metric_name = config['TRAINING']['save_metric']
        save_mode = config['TRAINING']['save_mode']
        best_metric = float('inf') if save_mode == "min" else float('-inf')
        logger.info(f"Model will be saved based on average {metric_name} ({save_mode})")


    config_dataset = load_config( "src/config/config.yaml")

    ## Load the Dataset 
    ds = bigearthnet_loader(
        path_dataset_lmdb=config_dataset["datasets"]["lmdb"],
        path_metadata_parquet=config_dataset["datasets"]["metadata_parquet"],
        path_metadata_snow_cloud_parquet=config_dataset["datasets"]["metadata_snow_cloud_parquet"],
        train= True
        )

    ## Load the DataModule
    dm = bigearthnet_DataModule(
        path_dataset_lmdb=config_dataset["datasets"]["lmdb"],
        path_metadata_parquet=config_dataset["datasets"]["metadata_parquet"],
        path_metadata_snow_cloud_parquet=config_dataset["datasets"]["metadata_snow_cloud_parquet"],
        batch_size=16
        )

        
    ## Create Train - Val and Test instances
    dm.setup(stage="fit")

    ## Loaders
    train_ds = dm.train_dataloader()
    val_ds = dm.val_dataloader()
    test_ds = dm.test_dataloader()



if __name__ == "__main__":
    main()
