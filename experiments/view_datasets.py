import os
import glob
import argparse
import fiftyone as fo

from tqdm import tqdm

def get_unique_labels(annotations: dict):
    # Get unique labels from annotations dictionary
    unique_labels = set()
    for annotation in annotations.values():
        for bbox_annotation in annotation:
            unique_labels.add(bbox_annotation["label"])
            
    # Sort labels
    unique_labels = list(unique_labels)
    unique_labels.sort()
    
    return unique_labels

def format_annotations(images_dir: str, labels_dir: str):
    # If images and annotations are in different directories
    images = glob.glob(f'{images_dir}/*.jpeg')
    labels = glob.glob(f'{labels_dir}/*.txt')
    
    annotations = {}
    print("Formatting annotations...")
    for label in tqdm(labels):
        base_name = os.path.splitext(os.path.basename(label))[0]
        image_file_path = os.path.join(images_dir, base_name + '.jpeg')
        
        with open(label, 'r') as file:
            lines = file.readlines()
            bbox_annotations = []
            for line in lines:
                # Assuming each line is in the format: label xmin ymin xmax ymax
                label, xmin, ymin, width, height = line.strip().split()
                xmin, ymin, width, height = map(float, (xmin, ymin, width, height))
                
                # Store bounding box values in a list
                bbox = [xmin - (width / 2), ymin - (height / 2), width, height]
                bbox_annotations.append({"bbox": bbox, "label": label})
        
        annotations[image_file_path] = bbox_annotations
    
    return annotations

def main(images_dir: str, labels_dir: str, dataset_name: str, class_names: list):
    
    annotations = format_annotations(images_dir=images_dir, labels_dir=labels_dir)
    
    # Create samples
    samples = []
    print("Creating samples...")
    for filepath in tqdm(annotations):
        sample = fo.Sample(filepath=filepath)
        
        detections = []
        for obj in annotations[filepath]:
            label = obj["label"]
            bounding_box = obj["bbox"]
            
            detections.append(
                fo.Detection(label=label, bounding_box=bounding_box)
            )
        
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)
        
    # Create dataset
    print("Creating dataset...")
    dataset = fo.Dataset(name=dataset_name)
    dataset.add_samples(samples)
    
    # Replace class names if needed
    unique_labels = get_unique_labels(annotations=annotations)
    mapping = {label: class_names[i] for i, label in enumerate(unique_labels)}
    view = dataset.map_labels("ground_truth", mapping)
    
    # Launch session
    session = fo.launch_app(dataset=view, remote=True, address="0.0.0.0")
    session.wait()

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', type=str, required=True,
                    help='Directory containing images')
    ap.add_argument('--labels_dir', type=str, required=True,
                    help='Directory containing YOLO annotations')
    ap.add_argument('--dataset_name', type=str, default='Temporary Name',
                    help='Name of the dataset')
    ap.add_argument('--class_names', type=str, nargs='+', default=[],
                    help='List of class names')
    args = ap.parse_args()
    
    main(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        dataset_name=args.dataset_name,
        class_names=args.class_names
    )
