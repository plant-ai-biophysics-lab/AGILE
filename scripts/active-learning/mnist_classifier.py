import argparse
import wandb
import glob
import lightning as pl

from src.data import MNISTDataModule
from src.models import LitClassifierModel
from src.sampling import UncertaintySampling
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.datasets import MNIST

def main(
    root_dir: str,
    logs_dir: str,
    batch_size: int,
    lr: float,
    epochs: int,
    method: str,
    rounds: int,
    chunk: float
):
    # set seed
    pl.seed_everything(42)
    
    # create sampling class
    us = UncertaintySampling(dataset=MNIST(root_dir, train=True))
    model = None
    
    for i in range(rounds):
        
        # define sampling method and model
        train_dataset, test_dataset = us.sample(chunk=chunk, method=method, model=model, batch_size=batch_size)
        if model is None:
            model = LitClassifierModel(num_classes=10, learning_rate=lr, hidden_dim=256)
        
        # create data module
        dm = MNISTDataModule(
            root_dir=root_dir, 
            batch_size=batch_size, 
            train_dataset=train_dataset,
            test_dataset=test_dataset
        )
        
        # create lightning trainer
        wandb_logger = WandbLogger(
            entity='paibl',
            project='active-learning',
            name=f'{args.name}_chunk-{i+1}',
            group=args.name,
            save_dir=args.logs_dir
        )
        checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(max_epochs=epochs, default_root_dir=logs_dir, logger=wandb_logger,
                            callbacks=[checkpoint_callback, lr_monitor])
        
        # run training
        trainer.fit(model, dm)
        
        # run test
        run_id = wandb_logger.version
        prj_name = wandb_logger.name
        ckpts = f'{logs_dir}/{prj_name}/{run_id}/checkpoints/*.ckpt'
        ckpt_path = glob.glob(ckpts)[0]
        trainer.test(ckpt_path=ckpt_path, datamodule=dm)
        model = LitClassifierModel.load_from_checkpoint(checkpoint_path=ckpt_path)
        
        wandb.finish() # finish wandb run
        i += 1 # update index

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the MNIST dataset.")
    ap.add_argument("--logs_dir", type=str, required=True,
                        help="Directory to save logs.")
    ap.add_argument("--name", type=str, required=True,
                        help="Root directory of the MNIST dataset.")
    ap.add_argument("--batch_size", type=int, default=32,
                        help="Number of samples per batch.")
    ap.add_argument("--lr", type=float, default=2e-4,
                        help="Rate at which to adjust model weights.")
    ap.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model.")
    ap.add_argument("--method", type=str, default="random",
                        help="Sampling method to use: [random, entropy, entropy_cluster].")
    ap.add_argument("--rounds", type=int, default=20,
                        help="Number of rounds of active learning.")
    ap.add_argument("--chunk", type=float, default=0.01,
                        help="Fraction of dataset to sample at each active learning round.")
    args = ap.parse_args()
    
    main(
        args.root_dir,
        args.logs_dir,
        args.batch_size,
        args.lr,
        args.epochs,
        args.method,
        args.rounds,
        args.chunk
    )