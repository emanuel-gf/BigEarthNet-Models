import wandb
from loguru import logger
import torch 
class WandbLogger:
    def __init__(self, config, result_dir):
        self.use_wandb = config['WANDB']['track']
        self.bands = config['model']['select_bands']

        if self.use_wandb:
            wandb.init(project=config['WANDB']['project_name'], config=config)
            wandb.run.name = result_dir["timestamp"]
            self.run = wandb.run
            logger.info(f"Initialized Weights & Biases run: {self.run.name}")
        else:
            self.run = None
            logger.info("Weights & Biases tracking is disabled.")

    def log_train(self, epoch, train_loss, val_loss, current_lr, train_metrics, val_metrics):
        if not self.use_wandb:
            return

        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        }

        # Add train metrics
        log_data.update({
            f"train/accuracy": train_metrics['accuracy'],
            f"train/f1_score": train_metrics['f1_score'],
            f"train/recall": train_metrics['recall'],
            f"train/precision": train_metrics['precision']
        })

        # Add validation metrics
        log_data.update({
            f"val/accuracy": val_metrics['accuracy'],
            f"val/f1_score": val_metrics['f1_score'],
            f"val/recall": val_metrics['recall'],
            f"val/precision": val_metrics['precision']
        })

        if "accuracy_per_class" in train_metrics:
            train_acc_per_class = train_metrics["accuracy_per_class"]
            # Convert tensor to list if needed
            if torch.is_tensor(train_acc_per_class):
                train_acc_per_class = train_acc_per_class.cpu().tolist()
            
            # Create bar chart for train per-class accuracy
            data = [[f"Class_{i}", acc] for i, acc in enumerate(train_acc_per_class)]
            table = wandb.Table(data=data, columns=["Class", "Accuracy"])
            log_data["train/accur_per_class"] = wandb.plot.bar(
                table, "Class", "Accuracy", 
                title="Training Accuracy per Class"
            )

        if "accuracy_per_class" in val_metrics:
            val_acc_per_class = val_metrics["accuracy_per_class"]
            # Convert tensor to list if needed
            if torch.is_tensor(val_acc_per_class):
                val_acc_per_class = val_acc_per_class.cpu().tolist()
            
            # Create bar chart for validation per-class accuracy
            data = [[f"Class_{i}", acc] for i, acc in enumerate(val_acc_per_class)]
            table = wandb.Table(data=data, columns=["Class", "Accuracy"])
            log_data["val/accur_per_class"] = wandb.plot.bar(
                table, "Class", "Accuracy", 
                title="Validation Accuracy per Class"
            )  
            
        wandb.log(log_data)

    def log_test(self, test_loss, test_metrics):
        if not self.use_wandb:
            return

        log_data = {
            "test_loss": test_loss,
        }

        log_data.update({
            f"test/accuracy": test_metrics['accuracy'],
            f"test/f1_score": test_metrics['f1_score'],
            f"test/recall": test_metrics['recall'],
            f"test/precision": test_metrics['precision']
        })

        # Handle per-class metrics for test
        if "accuracy_per_class" in test_metrics:
            test_acc_per_class = test_metrics["accuracy_per_class"]
            if torch.is_tensor(test_acc_per_class):
                test_acc_per_class = test_acc_per_class.cpu().tolist()
            
            data = [[f"Class_{i}", acc] for i, acc in enumerate(test_acc_per_class)]
            table = wandb.Table(data=data, columns=["Class", "Accuracy"])
            log_data["test/accur_per_class"] = wandb.plot.bar(
                table, "Class", "Accuracy", 
                title="Test Accuracy per Class"
            )

        if "f1_per_class" in test_metrics:
            test_f1_per_class = test_metrics["f1_per_class"]
            if torch.is_tensor(test_f1_per_class):
                test_f1_per_class = test_f1_per_class.cpu().tolist()
            
            data = [[f"Class_{i}", f1] for i, f1 in enumerate(test_f1_per_class)]
            table = wandb.Table(data=data, columns=["Class", "F1_Score"])
            log_data["test/f1_per_class"] = wandb.plot.bar(
                table, "Class", "F1_Score", 
                title="Test F1 Score per Class"
            )

        wandb.log(log_data)

    def save_model(self, model_path):
        if self.use_wandb:
            wandb.save(model_path)