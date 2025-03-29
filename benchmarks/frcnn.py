import os
import random
import cv2
import torch
import torchvision
import argparse
import shutil

import albumentations as A
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torchvision.ops as ops

class YoloDataset(Dataset):
    """
    Custom dataset to load images and YOLO-format labels.
    
    Expected:
      - image files (e.g., .jpg, .png) in image_dir
      - corresponding label files in label_dir with same basename and a .txt extension
      - each label file line: <class> <x_center> <y_center> <width> <height> (all normalized)
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        image = np.array(image)

        # Load label file (if exists)
        label_path = os.path.join(self.label_dir, os.path.splitext(image_filename)[0] + ".txt")
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_center, y_center, box_w, box_h = map(float, parts)

                    # Convert normalized coordinates to absolute pixel values
                    x_center *= orig_w
                    y_center *= orig_h
                    box_w *= orig_w
                    box_h *= orig_h

                    # Convert to [x_min, y_min, x_max, y_max]
                    x_min = x_center - box_w / 2
                    y_min = y_center - box_h / 2
                    x_max = x_center + box_w / 2
                    y_max = y_center + box_h / 2
                    
                    # if xmin or ymin is less than 5, add 1
                    if x_min < 5:
                        x_min += 1
                        x_max += 1
                    if y_min < 5:
                        y_min += 1
                        y_max += 1

                    # Clip coordinates to image dimensions
                    x_min = max(0, min(orig_w, x_min))
                    y_min = max(0, min(orig_h, y_min))
                    x_max = max(0, min(orig_w, x_max))
                    y_max = max(0, min(orig_h, y_max))

                    # Ensure positive width and height after clipping
                    if x_max <= x_min or y_max <= y_min:
                        continue  # skip invalid box

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls) + 1)  # shift class label by one if needed

        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=boxes, labels=labels)
            image = transformed["image"].float() / 255.0
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Handle images without bounding boxes
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor
        }

        return image, target

def collate_fn(batch):
    """Collate function needed for detection models which have varying number of targets per image."""
    return tuple(zip(*batch))

def draw_boxes(image_np, boxes, labels, scores=None, box_color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and labels on an image.
    
    Parameters:
      image_np: image as a numpy array (BGR format)
      boxes: numpy array of shape (N,4)
      labels: list or array of labels
      scores: (optional) list or array of scores (for predictions)
      box_color: color tuple for the rectangle (default green)
    """
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), box_color, thickness)
        label_text = f"{labels[i]}"
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
        cv2.putText(image_np, label_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return image_np

def prepare_symlink(save: Path, symlink_temp: str) -> None:
    if not (save / symlink_temp).exists():
        os.makedirs(save / symlink_temp)
    if not (save / symlink_temp / 'images').exists():
        os.makedirs(save / symlink_temp / 'images')
    if not (save / symlink_temp / 'images/train').exists():
        os.makedirs(save / symlink_temp / 'images/train')
    if not (save / symlink_temp / 'images/val').exists():
        os.makedirs(save / symlink_temp / 'images/val')
    if not (save / symlink_temp / 'images/test').exists():
        os.makedirs(save / symlink_temp / 'images/test')
    if not (save / symlink_temp / 'labels').exists():
        os.makedirs(save / symlink_temp / 'labels')
    if not (save / symlink_temp / 'labels/train').exists():
        os.makedirs(save / symlink_temp / 'labels/train')
    if not (save / symlink_temp / 'labels/val').exists():
        os.makedirs(save / symlink_temp / 'labels/val')
    if not (save / symlink_temp / 'labels/test').exists():
        os.makedirs(save / symlink_temp / 'labels/test')
        
def create_symlink(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source path {source} does not exist.")
    
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            item_target = target / item.name
            if item_target.exists():
                if item_target.is_symlink():
                    item_target.unlink()
                else:
                    raise FileExistsError(f"Target path {item_target} exists and is not a symlink.")
            os.symlink(item, item_target)
    else:
        if target.exists():
            if target.is_symlink():
                target.unlink()
            else:
                raise FileExistsError(f"Target path {target} exists and is not a symlink.")
        os.symlink(source, target)
        
def filter_outputs(outputs, score_threshold=0.3, iou_threshold=0.3):
    """
    Filters detections based on score threshold and applies Non-Maximum Suppression (NMS).
    
    Parameters:
        outputs (list or dict): Detection outputs from Faster R-CNN.
        score_threshold (float): Minimum confidence score to keep a detection.
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        Filtered outputs after applying score threshold and NMS.
    """
    filtered_outputs = []
    return_as_list = True
    if not isinstance(outputs, list):
        outputs = [outputs]
        return_as_list = False

    for output in outputs:
        scores = output["scores"]
        keep_indices = scores > score_threshold
        boxes = output["boxes"][keep_indices]
        scores = output["scores"][keep_indices]
        labels = output["labels"][keep_indices]

        if len(boxes) == 0:
            filtered_outputs.append({"boxes": [], "labels": [], "scores": []})
            continue

        nms_indices = ops.nms(boxes, scores, iou_threshold)
        filtered_output = {
            "boxes": boxes[nms_indices],
            "labels": labels[nms_indices],
            "scores": scores[nms_indices]
        }
        filtered_outputs.append(filtered_output)
    
    return filtered_outputs if return_as_list else filtered_outputs[0]

def main(
    source_images: Path,
    target_images: Path,
    generated_images: Path,
    save: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    symlink_temp: str,
    normal_augs: bool,
    lr: float,
    momentum: float,
    weight_decay: float,
    k_target: int  # new parameter for number of target samples to include or use
):
    # Determine source and target subset directories
    path_source_train = source_images / 'train' if (source_images / 'train').exists() else source_images
    path_target_val = target_images / 'val' if (target_images / 'val').exists() else target_images
    path_target_test = target_images / 'test' if (target_images / 'test').exists() else target_images

    # Prepare symlink library
    prepare_symlink(save, symlink_temp)

    # Symlink images and labels for training, validation, and test sets.
    if generated_images:
        create_symlink(generated_images, save / symlink_temp / 'images/train')
    else:
        create_symlink(path_source_train / 'images', save / symlink_temp / 'images/train')
    create_symlink(path_source_train / 'labels', save / symlink_temp / 'labels/train')
    create_symlink(path_target_val / 'images', save / symlink_temp / 'images/val')
    create_symlink(path_target_val / 'labels', save / symlink_temp / 'labels/val')
    create_symlink(path_target_test / 'images', save / symlink_temp / 'images/test')
    create_symlink(path_target_test / 'labels', save / symlink_temp / 'labels/test')

    # If generated images are provided, include k target images from the target set.
    if generated_images and k_target > 0:
        if (target_images / 'train').exists():
            target_train = target_images / 'train'
        elif (target_images / 'images').exists() and (target_images / 'labels').exists():
            target_train = target_images
        else:
            target_train = None
            print("Warning: Target training images not found in expected structure. Skipping addition of target images.")
        
        if target_train is not None:
            if (target_train / 'images').exists():
                target_train_images = target_train / 'images'
            else:
                target_train_images = target_train
            if (target_train / 'labels').exists():
                target_train_labels = target_train / 'labels'
            else:
                target_train_labels = target_train
            target_image_files = sorted([f for f in os.listdir(target_train_images) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if len(target_image_files) == 0:
                print("No images found in target training directory.")
            else:
                k = min(k_target, len(target_image_files))
                sampled_files = random.sample(target_image_files, k)
                for file in sampled_files:
                    src_image = target_train_images / file
                    dst_image = save / symlink_temp / 'images/train' / file
                    if not dst_image.exists():
                        os.symlink(src_image, dst_image)
                    label_file = os.path.splitext(file)[0] + ".txt"
                    src_label = target_train_labels / label_file
                    dst_label = save / symlink_temp / 'labels/train' / label_file
                    if os.path.exists(src_label) and not dst_label.exists():
                        os.symlink(src_label, dst_label)

    # New behavior: if generated images are NOT provided but k_target is specified,
    # then prune the training set to only use k_target random samples.
    if not generated_images and k_target > 0:
        print("Pruning training set to include only k_target samples...")
        train_images_dir = save / symlink_temp / 'images/train'
        train_labels_dir = save / symlink_temp / 'labels/train'
        train_files = sorted([f for f in os.listdir(train_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if len(train_files) > k_target:
            sampled_files = random.sample(train_files, k_target)
            for f in train_files:
                if f not in sampled_files:
                    os.remove(train_images_dir / f)
                    label_file = os.path.splitext(f)[0] + ".txt"
                    label_path = train_labels_dir / label_file
                    if os.path.exists(label_path):
                        os.remove(label_path)

    train_image_dir = save / symlink_temp / "images/train"
    train_label_dir = save / symlink_temp / "labels/train"
    val_image_dir = save / symlink_temp / "images/val"
    val_label_dir = save / symlink_temp / "labels/val"
    test_image_dir = save / symlink_temp / "images/test"
    test_label_dir = save / symlink_temp / "labels/test"

    # Create datasets and loaders with our simple resize & tensor transform
    orig_img = Image.open(os.path.join(test_image_dir, os.listdir(test_image_dir)[0]))
    orig_width, orig_height = orig_img.size
    gen_img = Image.open(os.path.join(train_image_dir, os.listdir(train_image_dir)[0]))
    gen_width, gen_height = gen_img.size
    
    transform_test = A.Compose([
        A.Resize(height=orig_height, width=orig_width, p=1.0),
        A.PadIfNeeded(
            min_height=orig_height, 
            min_width=orig_height, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=0,
            p=1.0
        ),
        A.Resize(height=image_size, width=image_size, p=1.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    if normal_augs:
        transform_train = transform_test
    else:
        transform_train = A.Compose([
            A.OneOf([
                A.RandomResizedCrop(size=(orig_height, orig_width), scale=(0.4, 0.8), ratio=(0.9, 1.1), p=0.5),
                A.PadIfNeeded(min_height=int(gen_height * 1.3), min_width=int(gen_width * 1.3),
                            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            ], p=1.0),
            A.Resize(height=orig_height, width=orig_width, p=1.0),
            A.PadIfNeeded(min_height=orig_height, min_width=orig_height,
                        border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            A.Resize(height=image_size, width=image_size, p=1.0),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
    train_dataset = YoloDataset(train_image_dir, train_label_dir, transform=transform_train)
    val_dataset   = YoloDataset(val_image_dir,   val_label_dir,   transform=transform_test)
    test_dataset  = YoloDataset(test_image_dir,  test_label_dir,  transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup pretrained Faster R-CNN with adjusted number of classes
    num_classes = 2  # e.g., 1 object class + background; adjust if more classes
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    metrics_history = []
    
    best_val_mAP = 0.0
    checkpoint_dir = save / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            valid_samples = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]
            if len(valid_samples) == 0:
                continue
            images, targets = zip(*valid_samples)
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}")

        # Evaluate on Validation Set
        print("Evaluating on validation set...")
        model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [img.to(device) for img in images]
                outputs = model(images)
                outputs = filter_outputs(outputs)
                for i in range(len(images)):
                    gt = {
                        "boxes": targets[i]["boxes"].cpu(),
                        "labels": targets[i]["labels"].cpu()
                    }
                    pred = {
                        "boxes": torch.as_tensor(outputs[i]["boxes"]).cpu(),
                        "labels": torch.as_tensor(outputs[i]["labels"]).cpu(),
                        "scores": torch.as_tensor(outputs[i]["scores"]).cpu()
                    }
                    all_gts.append(gt)
                    all_preds.append(pred)

        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError:
            raise ImportError("Please install torchmetrics: pip install torchmetrics")

        metric = MeanAveragePrecision()
        metric.update(all_preds, all_gts)
        map_dict = metric.compute()
        mAP_value = map_dict["map_50"].item() if isinstance(map_dict["map_50"], torch.Tensor) else map_dict["map_50"]
        print(f"Epoch [{epoch+1}/{epochs}], Validation mAP: {mAP_value:.4f}")
        
        if mAP_value > best_val_mAP:
            best_val_mAP = mAP_value
            best_checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mAP': mAP_value
            }, best_checkpoint_path)
            print(f"✅ Best model updated! Saved at {best_checkpoint_path}")

        epoch_metrics = {
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "val_mAP": mAP_value
        }

        if epoch == epochs - 1:
            print("Loading best checkpoint for final evaluation...")
            best_checkpoint_path = save / "checkpoints/best_model.pth"
            if best_checkpoint_path.exists():
                checkpoint = torch.load(best_checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                model.eval()
                print(f"Loaded best model from epoch {checkpoint['epoch']} with val mAP {checkpoint['val_mAP']:.4f}")
            else:
                print("⚠️ Warning: No best checkpoint found, using last trained model.")
                
            print("Evaluating on test set...")
            all_preds_test = []
            all_gts_test = []
            with torch.no_grad():
                for images, targets in tqdm(test_loader):
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    outputs = filter_outputs(outputs)
                    for i in range(len(images)):
                        gt_test = {
                            "boxes": targets[i]["boxes"].cpu(),
                            "labels": targets[i]["labels"].cpu()
                        }
                        pred_test = {
                            "boxes": torch.as_tensor(outputs[i]["boxes"]).cpu(),
                            "labels": torch.as_tensor(outputs[i]["labels"]).cpu(),
                            "scores": torch.as_tensor(outputs[i]["scores"]).cpu()
                        }
                        all_gts_test.append(gt_test)
                        all_preds_test.append(pred_test)

            metric_test = MeanAveragePrecision()
            metric_test.update(all_preds_test, all_gts_test)
            map_dict_test = metric_test.compute()
            mAP_test_value = map_dict_test["map_50"].item() if isinstance(map_dict_test["map_50"], torch.Tensor) else map_dict_test["map_50"]
            print(f"Test mAP: {mAP_test_value:.4f}")
            epoch_metrics["test_mAP"] = mAP_test_value

            # Save prediction samples for Validation Set
            sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
            os.makedirs(f"{save}/results/val_prediction_samples", exist_ok=True)
            for idx in sample_indices:
                image, target = val_dataset[idx]
                image_batch = image.to(device).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    output = model(image_batch)[0]
                    output = filter_outputs(output)
                img_np = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                boxes_gt = target["boxes"].cpu().numpy()
                labels_gt = target["labels"].cpu().numpy()
                img_with_gt = draw_boxes(img_np.copy(), boxes_gt, labels_gt, box_color=(0, 255, 0))

                boxes_pred = torch.as_tensor(output["boxes"]).cpu().numpy()
                labels_pred = torch.as_tensor(output["labels"]).cpu().numpy()
                scores_pred = torch.as_tensor(output["scores"]).cpu().numpy()
                img_with_preds = draw_boxes(img_with_gt, boxes_pred, labels_pred, scores=scores_pred, box_color=(0, 0, 255))
                cv2.imwrite(f"{save}/results/val_prediction_samples/pred_{idx}.jpg", img_with_preds)

            # Save prediction samples for Test Set
            sample_indices_test = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))
            os.makedirs(f"{save}/results/test_prediction_samples", exist_ok=True)
            for idx in sample_indices_test:
                image, target = test_dataset[idx]
                image_batch = image.to(device).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    output = model(image_batch)[0]
                    output = filter_outputs(output)
                img_np = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                boxes_gt = target["boxes"].cpu().numpy()
                labels_gt = target["labels"].cpu().numpy()
                img_with_gt = draw_boxes(img_np.copy(), boxes_gt, labels_gt, box_color=(0, 255, 0))

                boxes_pred = torch.as_tensor(output["boxes"]).cpu().numpy()
                labels_pred = torch.as_tensor(output["labels"]).cpu().numpy()
                scores_pred = torch.as_tensor(output["scores"]).cpu().numpy()
                img_with_preds = draw_boxes(img_with_gt, boxes_pred, labels_pred, scores=scores_pred, box_color=(0, 0, 255))
                cv2.imwrite(f"{save}/results/test_prediction_samples/pred_{idx}.jpg", img_with_preds)

        metrics_history.append(epoch_metrics)

    os.makedirs(f"{save}/results", exist_ok=True)
    df = pd.DataFrame(metrics_history)
    df.to_csv(f"{save}/results/metrics.csv", index=False)
    print("Results saved to results/metrics.csv")

    sample_indices_train = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
    os.makedirs(f"{save}/results/train_groundtruth_samples", exist_ok=True)
    for idx in sample_indices_train:
        image, target = train_dataset[idx]
        img_np = image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        img_with_gt = draw_boxes(img_np.copy(), boxes, labels, box_color=(0, 255, 0))
        cv2.imwrite(f"{save}/results/train_groundtruth_samples/train_gt_{idx}.jpg", img_with_gt)

    print("Training completed and results saved.")
    
    if (save / symlink_temp).exists():
        shutil.rmtree(save / symlink_temp)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-images', type=Path, required=True,
                    help='Path to training and validation dataset')
    ap.add_argument('--target-images', type=Path, required=True,
                    help='Path to target dataset')
    ap.add_argument('--generated-images', type=Path, required=False, default=None,
                    help='Path to generated images')
    ap.add_argument('--save', type=Path, required=True,
                    help='Path to save weights and results.')
    ap.add_argument('--image-size', type=int, default=800,
                    help='Image size for training.')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model.')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Batch size per epoch.')
    ap.add_argument('--symlink-temp', type=str, default='temp',
                    help='Name of the temporary symlink directory.')
    ap.add_argument('--run-name', type=str, default='temp',
                    help='Name of the run.')
    ap.add_argument('--normal-augs', action='store_true',
                    help='Use normal augmentations.')
    ap.add_argument('--lr', type=float, default=0.005,
                    help='Learning rate for training.')
    ap.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for training.')
    ap.add_argument('--weight-decay', type=float, default=0.0005,
                    help='Weight decay for training.')
    ap.add_argument('--k-target', type=int, default=0,
                    help='Number of target images to include in training if generated images is provided. '
                         'If generated images are not provided, only k samples from the source training set will be used.')
    args = ap.parse_args()
    
    save = args.save / args.run_name
    
    os.makedirs(save, exist_ok=True)
    with open(save / 'args.txt', 'w') as f:
        f.write(str(args))

    main(args.source_images, args.target_images, args.generated_images,
         save, args.image_size, args.epochs, args.batch_size,
         args.symlink_temp, args.normal_augs, args.lr,
         args.momentum, args.weight_decay, args.k_target)
