import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from collections import OrderedDict
from typing import List, Tuple
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.image_list import ImageList

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
        for boxes, scores, logits, image_shape in zip(pred_boxes_list, pred_scores_list, class_logits_list, original_image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(20, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            logits = logits[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            logits = logits.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.roi_heads.score_thresh)[0]
            boxes, scores, logits, labels = boxes[inds], scores[inds], logits[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, logits, labels = boxes[keep], scores[keep], logits[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.roi_heads.nms_thresh)
            boxes, scores, logits, labels = boxes[keep], scores[keep], logits[keep], labels[keep]

            all_logits.append(scores)

        return outputs, all_logits