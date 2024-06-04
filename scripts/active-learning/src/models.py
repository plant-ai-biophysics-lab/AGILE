import torchmetrics
import torch
import lightning as pl
import torch.nn.functional as F

from torch import nn

### SIMPLE CLASSIFIER ###
class LitClassifierModel(pl.LightningModule):
    
    def __init__(self, num_classes: int = 1, hidden_dim: int = 64, learning_rate: float = 2e-4):
        """Simple Classification model built with PyTorch Lightning.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1.
            hidden_dim (int, optional): Number of hidden layers. Defaults to 64.
            learning_rate (float, optional): Rate at which to adjust model weights. Defaults to 2e-4.
        """
        # update and save hyperparameters
        super().__init__()
        self.hparams.hidden_dim = hidden_dim
        self.hparams.learning_rate = learning_rate
        self.hparams.num_classes = num_classes
        self.save_hyperparameters()
        
        # layers
        self.l1 = nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = nn.Linear(self.hparams.hidden_dim, 10)
        
        # accuracy
        self.accuracy = torchmetrics.Accuracy(
            num_classes=self.hparams.num_classes, 
            average='macro', 
            task='multiclass')
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x
    
    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}
    
    def predict_step(self, batch):
        x = batch
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)