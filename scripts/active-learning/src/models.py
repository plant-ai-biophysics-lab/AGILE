import torchmetrics
import torch
import lightning as pl
import torch.nn.functional as F

from torch import nn
from src.util import CustomFasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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
    
### FASTER RCNN ###
class LitDetectorModel(pl.LightningModule):
    
    def __init__(
        self, 
        num_classes: int = 1, 
        learning_rate: float = 2e-4
    ):
        """Object Detection model built with PyTorch Lightning using Faster R-CNN.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1.
            learning_rate (float, optional): Rate at which to adjust model weights. Defaults to 2e-4.
        """
        super().__init__()
        
        # Define properties
        self.save_hyperparameters()
        self.hparams.lr = learning_rate
        self.hparams.num_classes = num_classes
        
        # Define the model
        # backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        backbone = fasterrcnn_resnet50_fpn().backbone
        self.model = CustomFasterRCNN(backbone, num_classes)
        
        # mAP calculation
        self.val_map_metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy')
        self.test_map_metric = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy')
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, images, targets=None):
        return self.model(images, targets)
        
    def training_step(self, batch):
        images, targets = batch
        targets = self.format_targets(targets)
        loss_dict = self.model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return losses
    
    def test_step(self, batch):
        images, targets = batch
        targets = self.format_targets(targets)
        outputs, _ = self.model(images)
        
        preds = [{k: v.detach() for k, v in t.items()} for t in outputs]
        
        # Wrap targets in a list of dictionaries
        formatted_targets = targets
        
        self.test_map_metric.update(preds, formatted_targets)
        self.test_outputs.append({'preds': preds, 'targets': formatted_targets})
        
        return outputs
    
    def on_test_epoch_end(self):
        if not self.test_outputs:
            mAP_result = {'map': torch.tensor(0.0)}
        else:
            mAP_result = self.test_map_metric.compute()
            self.test_map_metric.reset()

        # Log only the keys that contain "map"
        map_keys = {key: value for key, value in mAP_result.items() if 'map' in key}
        for key, value in map_keys.items():
            self.log(f'test_{key}', value, on_epoch=True, prog_bar=True, logger=True)
        
        self.test_outputs.clear()
        
    def predict_step(self, batch, dropout: bool = False):
        x = batch
        
        # activate dropout if true
        if dropout:
            self.model.train()
        else:
            self.model.eval()
            
        _, logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def format_targets(self, targets):
        """Convert the targets to the format expected by the model."""
        formatted_targets = []
        for boxes, labels in zip(targets['boxes'], targets['labels']):
            formatted_targets.append({
                'boxes': boxes,
                'labels': labels
            })
        return formatted_targets
