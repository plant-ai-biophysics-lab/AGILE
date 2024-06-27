import torch
import lightning as pl
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from torch import nn
from collections import OrderedDict
from typing import List, Tuple
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.image_list import ImageList

### CUSTOM FASTERRCNN CLASS ###
class CustomFasterRCNN(FasterRCNN):
    
    def __init__(self, backbone, num_classes):
        super(CustomFasterRCNN, self).__init__(backbone, num_classes)
        
    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
            outputs = super(CustomFasterRCNN, self).forward(images, targets)
            return outputs
        
        outputs = super(CustomFasterRCNN, self).forward(images, targets)
                    
        # check images
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images = ImageList(images, original_image_sizes)
        
        # extract features and metrics
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.rpn(images, features, targets)
        box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        
        # post process class logits
        device = class_logits.device
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        class_logits_list = class_logits.split(boxes_per_image, 0)
        
        all_logits = []
        for i, (boxes, scores, logits, image_shape) in enumerate(zip(pred_boxes_list, pred_scores_list, class_logits_list, original_image_sizes)):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(20, device=device)  # Assuming 19 classes + 1 background class
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            if boxes.size(1) > 1:
                boxes = boxes[:, 1:]
                scores = scores[:, 1:]
                labels = labels[:, 1:]
                logits = logits[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            logits = logits.repeat_interleave(19, dim=0)

            # remove low scoring boxes
            score_inds = torch.where(scores > self.roi_heads.score_thresh)[0]
            boxes, scores, labels, logits = boxes[score_inds], scores[score_inds], labels[score_inds], logits[score_inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, logits = boxes[keep], scores[keep], labels[keep], logits[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.roi_heads.nms_thresh)
            boxes, scores, labels, logits = boxes[keep], scores[keep], labels[keep], logits[keep]

            # append filtered logits to all_logits
            all_logits.append(logits.mean(dim=0))

        all_logits = torch.stack(all_logits)
        return outputs, all_logits
    
### MCDROPOUT MODULE ###
class MCDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)
    
### MODEL UPDATE FUNCTIONS ###
def update_model(
    model: pl.LightningModule,
    method: str,
    type: str = 'fasterrcnn'
):
    if method == 'BatchBALD' and type == 'fasterrcnn':
        model.roi_heads.box_head.fc6 = nn.Sequential(
            model.roi_heads.box_head.fc6,
            MCDropout(p=0.5)
        )
        model.roi_heads.box_head.fc7 = nn.Sequential(
            model.roi_heads.box_head.fc7,
            MCDropout(p=0.5)
        )
    
    return model