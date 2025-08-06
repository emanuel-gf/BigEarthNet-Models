import wandb
from loguru import logger

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
            f"train/precision": train_metrics['precision'],
            f"train/accur_per_class": train_metrics["accuracy_per_class"]
        })

        # Add validation metrics
        log_data.update({
            f"val/accuracy": val_metrics['precision'],
            f"val/f1_score": val_metrics['f1_score'],
            f"val/recall": val_metrics['recall'],
            f"val/precision": val_metrics['precision'],
            f"val/accur_per_class":val_metrics["accuracy_per_class"]
        })

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
            f"test/precision": test_metrics['precision'],
            f"test/accur_per_class": test_metrics["accuracy_per_class"],
            f"test/f1_per_class":test_metrics["f1_per_class"]
        })

        wandb.log(log_data)

    def save_model(self, model_path):
        if self.use_wandb:
            wandb.save(model_path)