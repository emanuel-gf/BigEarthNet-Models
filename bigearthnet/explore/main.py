import yaml
import os
from datetime import datetime
from loguru import logger  
import torch 
import torch.nn as nn 
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.utils import load_config
from src.data.loader import bigearthnet_loader, bigearthnet_DataModule
from src.utils.torch import seed_everything
from src.model_zoo.models import define_model_, define_model_scratch
from src.metrics.metrics import MultiLabelMetrics
from src.utils.wandb_logger import WandbLogger
from src.loader.reader import Dataset_BigEarthNet, Reader
from src.loader.reader import means_s2, stds_s2, get_list_means_std, get_right_dict, label_to_idx

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
def reader_(config_dataset, name_selected_bands):
    """ Create a class that handles the retrieval of tif files. 
    """
    reader = Reader(
        root_folder_path=config_dataset["datasets"]["root"],
        metadata_parquet_path = config_dataset["datasets"]["metadata_parquet"]
    )
    return reader 

def loader_dataset(train_test_split:str,reader, config, name_selected_bands, 
                    list_mean, list_std, label_to_idx, small_fraction=None):
    """
    Create a torch DataSet class that will iterate be passed on a datamodule
    """
    return Dataset_BigEarthNet(
        reader= reader,
        strip_bands = name_selected_bands,
        img_size=config["datasets"]["img_size"],
        upsample_mode=config["datasets"]["upsampling_method"],
        normalize =True,
        split_train_test= train_test_split,
        dict_one_hot= label_to_idx,
        small_fraction=small_fraction,
        transform= transforms.Compose([
            transforms.Normalize(mean=list_mean, std=list_std),
            transforms.Resize([224,224])
        ])
        )

def loader_dataloader(dataset,**kwargs):
    return DataLoader(
            dataset = dataset,
            **kwargs
        )
## It uses segmentation-models-torch to create the class that is by itself a nn.Torch
def build_model(config):


    # model = define_model_(
    #     model_name = config['model']['model_name'],
    #     num_classes = config['model']['num_classes'],
    #     input_channels =  config['model']['in_channels'],
    #     weights = config['model']['weight'],
    #     bands = config['model']['sentinel2_bands'],
    #     selected_channels = config['model']['select_bands']
    # )
    model = define_model_scratch(
        model_name = config['model']['model_name'],
        out_channel= config['model']['num_classes'],
        in_channel= config['model']['in_channels'],
        pretrained=config['model']['pretrained']
    )
    logger.info('Model created sucesfully')
    if config['model']['pretrained']:
        logger.info(model.pretrained_cfg)

    ## gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


def build_opt(model, config, pos_weight):
    optimizer_class = getattr(torch.optim, config['training']['optim'])

    ## weight decay
    weight_decay = config['training'].get('weight_decay', 0)

    if config['training']['optim'] == 'Adam':
        optimizer = optimizer_class(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(weight_decay)
        )
    elif config['training']['optim'] == 'SGD':
        logger.info(f"Adding Momentum to the given optimizer")
        optimizer = optimizer_class(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            momentum = config['training']['momentum_val'],
            weight_decay= float(weight_decay)
        )

    logger.info(f"Weight decay: {weight_decay}")

    scheduler = config['training']['scheduler']
    scheduler_class = None

    if scheduler:
        logger.info(f"scheduler type: {config['training']['scheduler_type']}")
        logger.info(f"scheduler factor: {config['training']['factor']}")
        
        lr_scheduler = getattr(torch.optim.lr_scheduler, config['training']['scheduler_type'])
        
        # For ReduceLROnPlateau, add patience parameter
        if config['training']['scheduler_type'] == 'ReduceLROnPlateau':
            patience = config['training'].get('patience', 10)  #patience of 10
            scheduler_class = lr_scheduler(
                optimizer, 
                mode='min', 
                factor=config['training']['factor'],
                patience=patience
            )
            logger.info(f"scheduler patience: {patience}")
        else:
            #change this if the scheduler is not ReduceLRonPlateaou
            scheduler_class = lr_scheduler(optimizer, factor=config['training']['factor'])

    ### Cross Entropy 
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion=nn.BCEWithLogitsLoss()

    return optimizer, criterion, scheduler, scheduler_class


def train_epoch(model, train_loader, optimizer, criterion, device, metrics_tracker, sensitivity):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for x_data, y_data in train_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            optimizer.zero_grad()
            outputs = model(x_data)
            loss = criterion(outputs, y_data.squeeze().float()) ## add loss 
            loss.backward() #compute gradient
            optimizer.step()

            out_sigmoid = torch.sigmoid(outputs)
            outputs_sens = (out_sigmoid>sensitivity).float()

            metrics_tracker.update(outputs_sens, y_data.squeeze())
            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    return train_loss / len(train_loader), metrics_tracker.compute()

def train_epoch_debug(model, train_loader, optimizer, criterion, device, metrics_tracker, sensitivity):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            x_data, y_data = x_data.to(device), y_data.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_data)

            loss = criterion(outputs, y_data.squeeze().float())

            # Debug loss and gradients
            if batch_idx == 0:
                print(f"Loss value: {loss.item()}")
                print(f"Model parameters require grad: {[p.requires_grad for p in model.parameters()][:3]}")
            

            loss.backward()
            # Check gradient flow
            if batch_idx == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f"Gradient norm: {total_norm}")

            optimizer.step()
            output_metrics = torch.sigmoid(outputs)
            outputs_sens = (output_metrics>sensitivity).float()
            
                        # Debug prints for first batch
            if batch_idx == 1:
                print(f"Input shape: {x_data.shape}")
                print(f"Label shape: {y_data.shape}")
                print(f"Label min/max: {y_data.min()}/{y_data.max()}")
                print(f"Label unique values: {torch.unique(y_data)}")
                print(f"Output min/max: {outputs.min()}/{outputs.max()}")
                #print(f"Metrics output - after sigmoid {output_metrics}")
                print(f"OUTPUT {outputs[:5]}")
                print(f"OUTPUT metrics {output_metrics[:5]}")
                print(f"After senstivity: ", outputs_sens)
                print(f"Output shape: {outputs.shape}")

            metrics_tracker.update(outputs_sens, y_data.squeeze())
            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    return train_loss / len(train_loader), metrics_tracker.compute()

def validate(model, val_loader, criterion, device, metrics_tracker, sensitivity):
    model.eval()
    metrics_tracker.reset()
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset), ncols=100, colour='#f4d160') as t:
            t.set_description("Validation")
            for x_data, y_data in val_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                outputs = model(x_data)
                loss = criterion(outputs, y_data.squeeze().float())

                output_metrics = torch.sigmoid(outputs)
                outputs_sens = (output_metrics>sensitivity).float()

                metrics_tracker.update(outputs_sens, y_data.squeeze())
                val_loss += loss.item()
                t.set_postfix(loss=loss.item())
                t.update(x_data.size(0))
                #metrics_tracker.update()

    return val_loss / len(val_loader), metrics_tracker.compute()


def test_model(model, test_loader, criterion, device, metrics_tracker, sensitivity):
    model.eval()
    metrics_tracker.reset()
    test_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(test_loader.dataset), ncols=100, colour='#cc99ff') as t:
            t.set_description("Testing")
            for x_data, y_data in test_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                outputs = model(x_data)
                loss = criterion(outputs, y_data.squeeze().float())
                out_sigmoid = torch.sigmoid(outputs)
                outputs_sens = (out_sigmoid>sensitivity).float()
                
                metrics_tracker.update(outputs_sens, y_data.squeeze())
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
    sensitivity = config['training']['sensitivity']
    logger.info(f"Sensitivity:{sensitivity}")
    num_epochs = config['training']['n_epoch']
    selected_bands = config['model']['select_bands']
    dict_sentinel2_bands =  config['model']['sentinel2_bands']
    name_selected_bands = [dict_sentinel2_bands[b] for b in selected_bands]
    small_fraction = config['training']['slice_of_training']

    logger.info(f"Number of selected bands: {len(selected_bands)}")
    logger.info(f"Selected Bands:{name_selected_bands}")

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

    ## Create the LMDB reader
    reader = reader_(config, name_selected_bands)
    logger.info("Reader READY")

    ## Select the dict containg the Std and Mean for each band
    mean_dict, std_dict = get_right_dict(s2_dict_mean=means_s2,
                                         s2_dict_std=stds_s2,
                                         upsampling_method=config['datasets']['upsampling_method'])
    
    ## select only the mean and std for the given selected_bands 
    list_mean, list_std = get_list_means_std(mean_dict=mean_dict,
                                             std_dict=std_dict,
                                            strip_bands=name_selected_bands
                                            )
    logger.info(f"List of average and stardard desviation ready: Mean= {list_mean} | Std {list_std} for bands |{selected_bands}")

    ## Create dataset instance
    dataset_train = loader_dataset('train',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, label_to_idx=label_to_idx, small_fraction=small_fraction)

    dataset_val = loader_dataset('validation',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, label_to_idx=label_to_idx, small_fraction=small_fraction)

    dataset_test = loader_dataset('test',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, label_to_idx=label_to_idx, small_fraction=small_fraction)
    
    logger.info(f'Size of train_dataset: {dataset_train.__len__()}')
    logger.info(f"Size of Val dataset: {dataset_val.__len__()}")
    logger.info(f"Size of Teste dataset: {dataset_test.__len__()}")
    
    ## Create the dataloader 
    train_dl = loader_dataloader(dataset_train,            
                                    batch_size= config['training']['batch_size'],
                                    shuffle = False,
                                    num_workers = 4,
                                    pin_memory = True)
    logger.info(f"Train dataset validate")

    test_dl = loader_dataloader(dataset_test,            
                                    batch_size= config['training']['batch_size'],
                                    shuffle = False,
                                    num_workers = 4,
                                    pin_memory = True)
    
    val_dl = loader_dataloader(dataset_val,            
                                    batch_size= config['training']['batch_size'],
                                    shuffle = False,
                                    num_workers = 4,
                                    pin_memory = True)
    
    ## Create the model 
    model, device = build_model(config)
    
    
    ## Calculate the positional weight
    if config['training']['positional_weight']==True:
        logger.warning(f"Calculate positional weight activate")
        pos_weight = dataset_train.calculate_unbalanced_df().to(device)
    else:
        pos_weight= None
    ## Define Optimizer, Scheduler and loss 
    optimizer, criterion, scheduler, scheduler_class = build_opt(model, config, pos_weight)

    ## Define Metrics Tracker 
    train_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"],
                                               threshold=config["training"]["sensitivity"]).to(device)
    val_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"], 
                                            threshold=config["training"]["sensitivity"]).to(device)
    test_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"],
                                              threshold=config["training"]["sensitivity"]).to(device)

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
        train_loss, train_metrics = train_epoch(model, train_dl, optimizer, criterion, device, train_metrics_tracker, sensitivity)
        val_loss, val_metrics = validate(model, val_dl, criterion, device, val_metrics_tracker, sensitivity)

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

    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pth')))
    test_loss, test_metrics = test_model(model, test_dl, criterion, device, test_metrics_tracker, sensitivity)

    wandb_logger.log_test(test_loss, test_metrics)

    # # save all metrics
    save_all_metrics(dict_metrics, test_metrics, selected_bands, num_epochs, metrics_path, train_losses, val_losses)



if __name__ == "__main__":
    main()
