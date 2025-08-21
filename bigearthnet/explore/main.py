import yaml
import os
from datetime import datetime
from loguru import logger  
import torch 
import random
import torch.nn as nn 
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from src.utils.utils import load_config
from src.utils.torch import seed_everything
from src.model_zoo.models import define_model_, define_model_scratch
from src.metrics.metrics import MultiLabelMetrics
from src.utils.wandb_logger import WandbLogger
from src.loader.reader import Dataset_BigEarthNet, Reader
from src.loader.reader import means_s2, stds_s2, get_list_means_std, get_right_dict


## Add code carbon
from codecarbon import EmissionsTracker, track_emissions

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
def reader_(config_dataset):
    """ Create a class that handles the retrieval of tif files. 
    """
    reader = Reader(
        root_folder_path=config_dataset["datasets"]["root"],
        metadata_parquet_path = config_dataset["datasets"]["metadata_parquet"]
    )
    return reader 

def loader_dataset(train_test_split,reader, config, name_selected_bands, 
                    list_mean, list_std, small_fraction=None,
                    ):
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
        small_fraction=small_fraction,
        transform= transforms.Compose([
            transforms.Normalize(mean=list_mean, std=list_std),
            transforms.Resize([224,224])
        ]),
        return_patch_id=False
        )

def loader_dataloader(dataset,**kwargs):
    return DataLoader(
            dataset = dataset,
            **kwargs
        )
## It uses segmentation-models-torch to create the class that is by itself a nn.Torch
def build_model(config):
    model = define_model_scratch(
        model_name = config['model']['model_name'],
        out_channel= config['model']['num_classes'],
        in_channel= len(config['model']['select_bands']),
        pretrained=config['model']['pretrained']
    )
    logger.info('Model created sucesfully')
    if config['model']['pretrained']==True:
        logger.success("Pretrained weights loaded successfully!!!")

    ## gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, device


def build_opt(model, config, pos_weight):
    optimizer_class = getattr(torch.optim, config['training']['optim'])  ## Adam

    ## weight decay
    optim_name = config['training'].get('optim', 'Adam')
    learning_rate = float(config['training'].get('learning_rate', 1e-4))
    weight_decay = float(config['training'].get('weight_decay', 1e-4))

    logger.info(f"Optimizer: {optim_name}, LR: {learning_rate}, Weight decay: {weight_decay}")
    
    try:
        optimizer_class = getattr(torch.optim, optim_name)
        
        if optim_name == 'Adam':
            # Adam with conservative defaults for stable training
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),  # Default betas
                eps=1e-8,  # Default eps
                amsgrad=False  # Can help with convergence in some cases
            )

        elif optim_name == 'SGD':
            momentum = float(config['training'].get('momentum_val', 0.9))
            nesterov = config['training'].get('nesterov', True)  # Nesterov momentum often helps
            
            if momentum <= 0 or momentum >= 1:
                logger.warning(f"Momentum {momentum} may be problematic. Setting to 0.9")
                momentum = 0.9
                
            logger.info(f"SGD with momentum: {momentum}, nesterov: {nesterov}")
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
        elif optim_name == 'AdamW':
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),  # Default betas
                eps=1e-8,  # Default eps
                amsgrad=False  # Can help with convergence in some cases
            )
    except AttributeError:
        raise ValueError(f"Optimizer '{optim_name}' not found in torch.optim")

    scheduler = config['training']['scheduler']
    scheduler_class = None

    if scheduler:
        logger.info(f"scheduler type: {config['training']['scheduler_type']}")
        logger.info(f"scheduler factor: {config['training']['factor']}")

        scheduler_type = config['training'].get('scheduler_type', 'ReduceLROnPlateau')
        factor = float(config['training'].get('factor', 0.5))
        
        
        try:
            lr_scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)
            
            if scheduler_type == 'ReduceLROnPlateau':
                patience = int(config['training'].get('patience', 5))
                min_lr = float(config['training'].get('min_lr', 1e-7))
                
                scheduler_class = lr_scheduler(
                    optimizer, 
                    mode='min', 
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr,
                    threshold = 1e-3
                )
                logger.info(f"ReduceLROnPlateau: patience={patience}, min_lr={min_lr}")


        except AttributeError:
            logger.error(f"Scheduler '{scheduler_type}' not found in torch.optim.lr_scheduler")
        except Exception as e:
            logger.error(f"Error creating scheduler: {e}")

    ### Cross Entropy 
    try:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion=nn.BCEWithLogitsLoss()
    except Exception as e:
        logger.error(f"Error creating loss function: {e}")
        # Fallback to basic BCE loss
        criterion = nn.BCEWithLogitsLoss()

    return optimizer, criterion, scheduler, scheduler_class


def train_epoch(model, train_loader, optimizer, criterion, device, metrics_tracker, sensitivity):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for x_data, y_data in train_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)

            if len(y_data.shape) == 3 and y_data.shape[1] == 1:
                            y_data = y_data.squeeze(1)  # [batch_size, num_classes]

            optimizer.zero_grad()
            outputs = model(x_data)

            loss = criterion(outputs, y_data.float()) ## add loss 
            loss.backward() #compute gradient

            # Gradient clipping to prevent exploding gradients
            ##torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() 
            
            with torch.no_grad():
                out_sigmoid = torch.sigmoid(outputs)
                outputs_sens = (out_sigmoid>sensitivity).float()
                metrics_tracker.update(outputs_sens, y_data)
            
            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    return train_loss / len(train_loader), metrics_tracker.compute()

def train_epoch_debug(model, train_loader, optimizer, criterion, device, metrics_tracker, sensitivity,mlb):
    model.train()
    metrics_tracker.reset()
    train_loss = 0.0

    with tqdm(total=len(train_loader.dataset), ncols=100, colour='#3eedc4') as t:
        t.set_description("Training")
        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            x_data, y_data = x_data.to(device), y_data.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_data)   

            ## only squeeze if necessary
            if len(y_data.shape) == 3 and y_data.shape[1] == 1:
                y_data = y_data.squeeze(1)  #batch_size, num_classes

            loss = criterion(outputs, y_data.float())

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

            with torch.no_grad():
                output_metrics = torch.sigmoid(outputs)
                outputs_sens = (output_metrics>sensitivity).float()
                metrics_tracker.update(outputs_sens, y_data)

            if batch_idx == 0:
                print(f"Input shape: {x_data.shape}")
                print(f"Label shape: {y_data.shape}")
                print(f"Label min/max: {y_data.min()}/{y_data.max()}")
                print(f"Label unique values: {torch.unique(y_data)}")
                print(f"Output min/max: {outputs.min()}/{outputs.max()}")
                print(f"Output Sigmoid min/max: {output_metrics.min()}/{output_metrics.max()}")
                ##print(f"OUTPUT {outputs[:2]}")
                print(f"OUTPUT Sigmoid metrics {output_metrics[:2]}")
                print(f"After senstivity: ", outputs_sens[:2])
                tensor_to_numpy = outputs_sens.clone().detach().cpu().numpy()
                n1 = tensor_to_numpy.astype(int)
                print("class label")
                for i in n1[:10]:
                    print(f"{mlb.inverse_transform(i.reshape(1,-1))}") 
                print("Ground truth")
                for i in y_data.clone().detach().cpu().numpy().astype(int)[:10]:
                    print(f"mlb gt: {mlb.inverse_transform(i.reshape(1,-1))}")
                print(f"Output shape: {outputs.shape}")

            train_loss += loss.item()
            t.set_postfix(loss=loss.item())
            t.update(x_data.size(0))

    

    return train_loss / len(train_loader), metrics_tracker.compute()


def validate(model, val_loader, criterion, device, metrics_tracker, sensitivity,mlb):
    model.eval()
    metrics_tracker.reset()
    val_loss = 0.0

    with torch.no_grad():
        with tqdm(total=len(val_loader.dataset), ncols=100, colour='#f4d160') as t:
            t.set_description("Validation")
            for batch_idx, (x_data, y_data) in enumerate(val_loader):
                x_data, y_data = x_data.to(device), y_data.to(device)

                outputs = model(x_data)

                ## only squeeze if necessary
                if len(y_data.shape) == 3 and y_data.shape[1] == 1:
                    y_data = y_data.squeeze(1)  #batch_size, num_classes

                loss = criterion(outputs, y_data.float())

                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    continue

                output_metrics = torch.sigmoid(outputs)
                outputs_sens = (output_metrics>sensitivity).float()

                if batch_idx == 0:
                    print("---"*30)
                    print(f"Input shape: {x_data.shape}")
                    print(f"Label shape: {y_data.shape}")
                    print(f"Label min/max: {y_data.min()}/{y_data.max()}")
                    print(f"Label unique values: {torch.unique(y_data)}")
                    print(f"Output min/max: {outputs.min()}/{outputs.max()}")
                    print(f"Output Sigmoid min/max: {output_metrics.min()}/{output_metrics.max()}")
                    ##print(f"OUTPUT {outputs[:2]}")
                    print(f"OUTPUT Sigmoid metrics {output_metrics[:2]}")
                    print(f"After senstivity: ", outputs_sens[:2])
                    tensor_to_numpy = outputs_sens.clone().detach().cpu().numpy()
                    n1 = tensor_to_numpy.astype(int)
                    print("class label")
                    for i in n1[:10]:
                        print(f"{mlb.inverse_transform(i.reshape(1,-1))}") 
                    print("Ground truth")
                    for i in y_data.clone().detach().cpu().numpy().astype(int)[:10]:
                        print(f"mlb gt: {mlb.inverse_transform(i.reshape(1,-1))}")
                    print(f"Output shape: {outputs.shape}")

                metrics_tracker.update(outputs_sens, y_data)
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

                ## only squeeze if necessary
                if len(y_data.shape) == 3 and y_data.shape[1] == 1:
                    y_data = y_data.squeeze(1)  #batch_size, num_classes

                loss = criterion(outputs, y_data.float())
                out_sigmoid = torch.sigmoid(outputs)
                outputs_sens = (out_sigmoid>sensitivity).float()
                
                metrics_tracker.update(outputs_sens, y_data)
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


## track emission 
#@track_emissions(save_to_api=True,logging_logger=False, log_level='critical')
def main()->None:

    ## Create out dirs
    paths = create_result_dirs()
    log_path = paths['log_path']
    checkpoint_path = paths['checkpoint_path']
    metrics_path = paths['metrics_path']

    ## Load yaml file with configs
    config = load_config("src/config/config.yaml")

    ## MLB 
    df_parquet = pd.read_parquet(config['datasets']['metadata_parquet'])

    sensitivity = config['training']['sensitivity']
    logger.info(f"Sensitivity:{sensitivity}")
    num_epochs = config['training']['n_epoch']
    selected_bands = config['model']['select_bands']
    dict_sentinel2_bands =  config['model']['sentinel2_bands']
    name_selected_bands = [dict_sentinel2_bands[b] for b in selected_bands]

    logger.info(f"Number of selected bands: {len(selected_bands)}")
    logger.info(f"Selected Bands:{name_selected_bands}")

    small_fraction = config['training']['slice_of_training']
    if int(small_fraction)==int(1.0):
        small_fraction = None
    else: 
        logger.info(f"Fraction of training data: {small_fraction}")

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
    if config['training']['positional_weight']==True:
        ps_str = 'PS'
    else:
        ps_str = '-'
    save_config_to_log(config, paths['result_dir'])
    name_run = f"{str(len(selected_bands))}B" +f"-{sensitivity}sens" + f"-{str(config['training']['learning_rate'])}LR"+ f"{str(config['training']['weight_decay'])}WD"+f"-{ps_str}"##+##f"{str(paths['result_dir']['timestamp'])}"
    logger.warning(f"Run name: {name_run}")
    # set up weight and bias to track experiment
    wandb_logger = WandbLogger(config=config, result_dir=paths, name_run = name_run)

    ## Create the Reader
    reader = reader_(config)
    logger.success("Reader READY")

    ## Select the dict containg the Std and Mean for each band
    mean_dict, std_dict = get_right_dict(s2_dict_mean=means_s2,
                                         s2_dict_std=stds_s2,
                                         upsampling_method=config['datasets']['upsampling_method'])
    
    ## select only the mean and std for the given selected_bands 
    list_mean, list_std = get_list_means_std(mean_dict=mean_dict,
                                             std_dict=std_dict,
                                            strip_bands=name_selected_bands
                                            )
    # list_mean =[0.485, 0.456, 0.406]  # RGB order
    # list_std=[0.229, 0.224, 0.225]   # RGB order
    
    logger.info(f"List of average and stardard desviation ready: Mean= {list_mean} | Std {list_std} for bands |{selected_bands}")

    ## Create dataset instance
    dataset_train = loader_dataset('train',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, small_fraction=small_fraction

                                            )

    dataset_val = loader_dataset('validation',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, small_fraction=small_fraction)

    dataset_test = loader_dataset('test',reader=reader, config=config,
                                            name_selected_bands=name_selected_bands, list_mean=list_mean,
                                            list_std=list_std, small_fraction=small_fraction)
    
    logger.info(f'Size of train_dataset: {dataset_train.__len__()}')
    logger.info(f"Size of Val dataset: {dataset_val.__len__()}")
    logger.info(f"Size of Teste dataset: {dataset_test.__len__()}")
    
    ## Create the dataloader 
    train_dl = loader_dataloader(dataset_train,            
                                    batch_size= config['training']['batch_size'],
                                    shuffle = True,
                                    num_workers = 4,
                                    pin_memory = True)

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
    train_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"]).to(device)
    val_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"]).to(device)
    test_metrics_tracker = MultiLabelMetrics(num_classes=config["model"]["num_classes"]).to(device)

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
    save_model= True
    train_losses = []
    val_losses = []

    mlb = dataset_train.return_mlb() ## Get the multibinarizer instance for labels 
    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch_debug(model, train_dl, optimizer, criterion, device, train_metrics_tracker, sensitivity, mlb)
        val_loss, val_metrics = validate(model, val_dl, criterion, device, val_metrics_tracker, sensitivity,mlb)

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

        wandb_logger.log_train(epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics)

        save_model = True

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
            #wandb_logger.save_model(model_path)
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
