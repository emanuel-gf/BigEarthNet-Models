import yaml
import os
from datetime import datetime
from loguru import logger  
import torch 
import torch.nn as nn 
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

from src.utils.utils import load_config
from src.data.loader import bigearthnet_loader, bigearthnet_DataModule
from src.utils.torch import seed_everything
from src.model_zoo.models import define_model_
from src.metrics.metrics import MultiLabelMetrics
from src.utils.wandb_logger import WandbLogger


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

## Seed everything
def setup_environment(config, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.add(log_path, rotation="10 MB")
    seed_everything(seed=config['training']['seed'])


## Save logs 
def save_config_to_log(config, log_dir, filename="config.yaml"):
    os.makedirs(log_dir, exist_ok=True)
    config_path = os.path.join(log_dir, filename)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info(f"Saved config to {config_path}")


## Build up model by definition 
## It uses segmentation-models-torch to create the class that is by itself a nn.Torch
def build_model(config):


    model = define_model_(
        model_name = config['model']['model_name'],
        num_classes = config['model']['num_classes'],
        input_channels =  config['model']['in_channels'],
        weights = config['model']['weight'],
        bands = config['model']['sentinel2_bands'],
        selected_channels = config['model']['select_bands']
    )
    

    ## gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


def build_opt(model, config):
    optimizer_class = getattr(torch.optim, config['training']['optim'])

    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
    )
    scheduler = config['training']['scheduler']
    if scheduler:
        logger.info(f"scheduler type: {config['training']['scheduler_type']}")
        logger.info(f"scheduler factor: {config['training']['factor']}")
        lr_scheduler = getattr(torch.optim.lr_scheduler, config['training']['scheduler_type'])
        scheduler_class = lr_scheduler(optimizer, mode='min',factor=config['training']['factor'])
    else:
        scheduler_class = None

    ### Cross Entropy 
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion, scheduler, scheduler_class


def train_epoch(model, train_loader, optimizer, criterion, device, metrics_tracker):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for x_data, y_data in train_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            optimizer.zero_grad()
            outputs = model(x_data)
            loss = criterion(outputs, y_data.float()) ## add loss 
            loss.backward() #compute gradient
            optimizer.step()

            metrics_tracker.update(outputs, y_data)
            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    return train_loss / len(train_loader), metrics_tracker.compute()


def validate(model, val_loader, criterion, device, metrics_tracker):
    model.eval()
    metrics_tracker.reset()
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset), ncols=100, colour='#f4d160') as t:
            t.set_description("Validation")
            for x_data, y_data in val_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                outputs = model(x_data)
                loss = criterion(outputs, y_data)
                metrics_tracker.update(outputs, y_data)
                val_loss += loss.item()
                t.set_postfix(loss=loss.item())
                t.update(x_data.size(0))
                #metrics_tracker.update()

    return val_loss / len(val_loader), metrics_tracker.compute()

def run_test(model, test_loader, criterion, device, metrics_tracker, checkpoint_path, wandb_logger=None):
    """
    Runs test evaluation on the best saved model.

    Args:
        model: your PyTorch model instance
        test_loader: test DataLoader
        criterion: loss function
        device: torch.device
        metrics_tracker: metric calculator instance
        checkpoint_path: path to load best model weights from
        wandb_logger: optional, for logging test results to wandb

    Returns:
        test_loss: average test loss
        test_metrics: dict of test metrics (aggregated)
    """
    import torch

    # Load best weights
    best_model_path = os.path.join(checkpoint_path, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    test_losses = []
    metrics_tracker.reset()

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())

            metrics_tracker.update(outputs, targets)

    test_loss = sum(test_losses) / len(test_losses)
    test_metrics = metrics_tracker.compute()
    metrics_tracker.reset()

    if wandb_logger:
        wandb_logger.log_test(test_loss, test_metrics)

    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test Metrics: {test_metrics}")

    return test_loss, test_metrics

def test_model(model, test_loader, criterion, device, metrics_tracker):
    model.eval()
    metrics_tracker.reset()
    test_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(test_loader.dataset), ncols=100, colour='#cc99ff') as t:
            t.set_description("Testing")
            for x_data, y_data in test_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                outputs = model(x_data)
                loss = criterion(outputs, y_data)
                metrics_tracker.update(outputs, y_data)
                test_loss += loss.item()
                t.set_postfix(loss=loss.item())
                t.update(x_data.size(0))

    return test_loss / len(test_loader), metrics_tracker.compute()


def save_all_metrics(dict_metrics, test_metrics, bands, num_epochs, save_path, train_losses, val_losses):
    os.makedirs(save_path, exist_ok=True)

    # Save train/val metrics per epoch
    metrics_to_save = ['accuracy', 'f1_score', 'precision', 'recall']
    phases = ['train', 'val']
    
    for metric in metrics_to_save:
        df_data = {'epoch': list(range(num_epochs))}
        for phase in phases:
            key = f"{phase}_{metric}"
            if key in dict_metrics:
                df_data[key] = dict_metrics[key]
            else:
                logger.warning(f"Key {key} not found in dict_metrics.")
        df = pd.DataFrame(df_data)
        file_path = os.path.join(save_path, f"{metric}_metrics.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {metric} metrics to {file_path}")

    # Save test metrics summary if available - Fix it, eliminate the bands
    if test_metrics:
        test_summary = {
        }
        for metric in metrics_to_save:
            test_summary[metric] = [test_metrics.get(metric, None)]
        df_test = pd.DataFrame(test_summary)
        test_path = os.path.join(save_path, "test_metrics_summary.csv")
        df_test.to_csv(test_path, index=False)
        logger.info(f"Saved test metrics summary to {test_path}")

    # Save train/val losses
    df_loss = pd.DataFrame({
        'epoch': list(range(num_epochs)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_path = os.path.join(save_path, "losses.csv")
    df_loss.to_csv(loss_path, index=False)
    logger.info(f"Saved train/val losses to {loss_path}")

def main()->None:

    config_dataset = load_config( "src/config/config.yaml")
    ## Create out dirs
    paths = create_result_dirs()
    log_path = paths['log_path']
    checkpoint_path = paths['checkpoint_path']
    metrics_path = paths['metrics_path']

    ## Load yaml file with configs
    config = load_config("src/config/config.yaml")

    #bands = config['DATASET']['bands']
    num_epochs = config['training']['n_epoch']
    selected_bands = config['model']['select_bands']
    len_img_size_channel = len(selected_bands)
    # Initialize best metrics at the beginning of training
    if config['training']['save_strategy'] == "loss":
        best_metric = float('inf')  # For loss, lower is better
        logger.info("Model will be saved based on validation loss")
    else:  # metric-based saving
        metric_name = config['training']['save_metric']
        save_mode = config['training']['save_mode']
        best_metric = float('inf') if save_mode == "min" else float('-inf')
        logger.info(f"Model will be saved based on average {metric_name} ({save_mode})")


    ## SETUP env
    setup_environment(config,log_path)
    save_config_to_log(config, paths['result_dir'])
    # set up weight and bias to track experiment
    wandb_logger = WandbLogger(config=config, result_dir=paths)

    ## Load the DataModule - Create DataAugmentation by Default
    dm = bigearthnet_DataModule(
        path_dataset_lmdb=config_dataset["datasets"]["lmdb"],
        path_metadata_parquet=config_dataset["datasets"]["metadata_parquet"],
        path_metadata_snow_cloud_parquet=config_dataset["datasets"]["metadata_snow_cloud_parquet"],
        batch_size= config['training']['batch_size'],
        img_size = (len_img_size_channel,120,120), ## EarthNet is 120x120,
        shuffle=False,
        #max_len= 80,  ## test if it is working
        train_transforms = transforms.Compose([
                                transforms.Resize((224, 224))  # Direct resize to expected size
                                ]),
        eval_transforms = transforms.Compose([
                                    transforms.Resize((224, 224))  # Direct resize to expected size
                                ])
        )

    ## Train - Val and Test instances
    ## This instance populates the dm object. 
    dm.setup(stage="fit")

    ## Loaders
    train_ds = dm.train_dataloader()
    val_ds = dm.val_dataloader()
    logger.info(f'Size of train_dataset: {len(train_ds)}')
    logger.info(f"Size of Val dataset: {len(val_ds)}")

    ## Create the model 
    model, device = build_model(config)

    ## Define Optimizer, Scheduler and loss 
    optimizer, criterion, scheduler, scheduler_class = build_opt(model, config)

    ## Define Metrics Tracker 
    train_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"], threshold=0.6).to(device)
    val_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"], threshold=0.6).to(device)
    test_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"], threshold=0.6).to(device)

    # test_metrics_tracker = MultiClasses(num_classes=config["model"]["num_classes"])

    dict_metrics = {
        'train_accuracy': [],
        'train_f1_score': [],
        'train_precision': [],
        'train_recall': [],
        'train_acc_per_class':[],
        'val_accuracy': [],
        'val_f1_score': [],
        'val_precision': [],
        'val_recall': [],
        'val_acc_per_class':[]

    }

    best_val_loss=float('inf')
    save_model= False
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(model, train_ds, optimizer, criterion, device, train_metrics_tracker)
        val_loss, val_metrics = validate(model, val_ds, criterion, device, val_metrics_tracker)

        ## pass the scheduler for each step
        if scheduler:
            scheduler_class.step(val_loss)
        
        # check and modified if necessary
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.8f}")
        logger.info(f"Epoch {epoch+1}: Train Loss= {train_loss:.6f}, Val Loss={val_loss:.6f}")

        ## Add everything to the dict_metrics
        dict_metrics['train_accuracy'].append(train_metrics['accuracy'])
        dict_metrics['train_f1_score'].append(train_metrics['f1_score'])
        dict_metrics['train_precision'].append(train_metrics['precision'])
        dict_metrics['train_recall'].append(train_metrics['recall'])

        dict_metrics['val_accuracy'].append(val_metrics['accuracy'])
        dict_metrics['val_f1_score'].append(val_metrics['f1_score'])
        dict_metrics['val_precision'].append(val_metrics['precision'])
        dict_metrics['val_recall'].append(val_metrics['recall'])
        dict_metrics['train_acc_per_class'].append(train_metrics['accuracy_per_class'])
        dict_metrics['val_acc_per_class'].append(val_metrics['accuracy_per_class'])
        wandb_logger.log_train(epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics)

        save_model = False

        if config["training"]['save_strategy']=="loss":
            if val_loss < best_metric:
                best_metric = val_loss
                save_model = True
                save_message = f"Best model saved at epoch {epoch+1} with Val Loss: { best_metric:.6f}"
        else: 
            metric_name = config['training']['save_metric']
            save_mode = config['training']['save_mode']
            avg_metric = val_metrics.get(metric_name, 0.0) 

            if (save_mode == "min" and avg_metric < best_metric) or \
            (save_mode == "max" and avg_metric > best_metric):
                best_metric = avg_metric
                save_model = True
                save_message = f"Best model saved at epoch {epoch+1} with avg {metric_name}: {best_metric:.6f}"
        
         # Save model if criteria met
        if save_model:
            model_path = os.path.join(checkpoint_path, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            wandb_logger.save_model(model_path)
            logger.info(save_message)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    

    ## Populate the dataset with the test 
    dm.setup('test')
    test_ds = dm.test_dataloader()
    logger.info(f"Size of Test Dataset: {test_ds}")
    if test_ds is not None:
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth')))
        test_loss, test_metrics = test_model(model, test_ds, criterion, device, test_metrics_tracker)

        wandb_logger.log_test(test_loss, test_metrics)

    # # save all metrics
        save_all_metrics(dict_metrics, test_metrics, selected_bands, num_epochs, metrics_path, train_losses, val_losses)
    else:
        print(f'Issues saving the metrics. Please retry')


if __name__ == "__main__":
    main()
