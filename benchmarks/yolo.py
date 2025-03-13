import argparse
import yaml
import sys
import os
import random
import string
import shutil

from ultralytics import YOLO
from pathlib import Path
from typing import List
     
def edit_data(
    data: Path,
    new_path: dict
) -> None:
    """
    Edits existing data.yaml file with new path and labels.
    
    Args:
        data (Path): Path to data.yaml file.
        new_path (dict): dictionary with new values to update.
    """
    
    try:
        with open(str(data), 'r') as file:
            d = yaml.safe_load(file)
            
        # update path value in yaml file
        for key, value in new_path.items():
            keys = key.split('.')
            temp_d = d
            for k in keys[:-1]:
                if k not in temp_d:
                    raise KeyError(f"Key '{key}' not found in the YAML file.")
                temp_d = temp_d[k]
            temp_d[keys[-1]] = value
            
        print(f"YAML file '{data}' has been updated.")
            
        # save new yaml file
        with open(data, 'w') as file:
            yaml.dump(d, file, default_flow_style=False)
        
    except Exception as e:
        print(f'An error occured: {e}')

def load_yaml_config(
    config_file: Path
) -> dict:
    """
    Loads training configuration file into dictionary
    
    Args:
        config_file (Path): Path to config.yaml file.
    """
    
    # check if config file exists
    if not config_file.exists():
        print(f'Config file {config_file} not found.')
        sys.exit(1)
    
    # opens config.yaml file and returns a dictionary
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_hash(length=6):
    """Generate a hash for model where it starts with the trait followed by a random string of characters.

    Args:
        trait (str): trait to be analyzed (plant, flower, pod, etc.)
        length (int, optional): Length for random sequence. Defaults to 5.
    """
    random_sequence = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    hash_id = f"{random_sequence}"
    return hash_id
    
def prepare_symlink(
    save: Path,
    symlink_temp: str,
) -> None:
    
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
                    item_target.unlink()  # Remove the existing symlink
                else:
                    raise FileExistsError(f"Target path {item_target} exists and is not a symlink.")
            os.symlink(item, item_target)
    else:
        if target.exists():
            if target.is_symlink():
                target.unlink()  # Remove the existing symlink
            else:
                raise FileExistsError(f"Target path {target} exists and is not a symlink.")
        os.symlink(source, target)

def main(
    pretrained: Path,
    source_images: Path,
    target_images: Path,
    generated_images: Path,
    config_path: Path,
    data: Path,
    save: Path,
    labels: List[str],
    image_size: int,
    epochs: int,
    batch_size: int,
    run_name: str,
    symlink_temp: str,
    model_type: str
) -> None:
    
    try:
        
        # get train, val and test images/labels
        path_source_train = source_images / 'train' if (source_images / 'train').exists() else source_images
        path_target_val = target_images / 'val' if (target_images / 'val').exists() else target_images
        path_target_test = target_images / 'test' if (target_images / 'test').exists() else target_images
        
        # prepare symlink library
        prepare_symlink(save, symlink_temp)
        
        # symlink images and labels
        if generated_images:
            create_symlink(generated_images, save / symlink_temp / 'images/train')
        else:
            create_symlink(path_source_train / 'images', save / symlink_temp / 'images/train')
        create_symlink(path_source_train / 'labels', save / symlink_temp / 'labels/train')
        create_symlink(path_target_val / 'images', save / symlink_temp / 'images/val')
        create_symlink(path_target_val / 'labels', save / symlink_temp / 'labels/val')
        create_symlink(path_target_test / 'images', save / symlink_temp / 'images/test')
        create_symlink(path_target_test / 'labels', save / symlink_temp / 'labels/test')

        if model_type == 'yolo':
            
            # edit data.yaml file with update path to images
            new_path: dict = {
                'path': str(save / symlink_temp),
                'names': labels,
                'nc': len(labels),
                'train': 'images/train',
                'val': 'images/val'
            }
            edit_data(data, new_path)
            
            # initialize project and update config
            config = load_yaml_config(config_file=config_path)
            version = generate_hash()
            while (save / run_name / Path(f'{version}')).exists():
                version = generate_hash()
            name = f'{version}'
            new_values: dict = {
                'batch': batch_size,
                'epochs': epochs,
                'imgsz': image_size,
                'project': f'{save}/{run_name}_{version}',
                'name': name,
                'data': str(data),
                'model': str(pretrained)
            }
            config.update(new_values)
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            
            # load pretrained model
            model = YOLO(pretrained)

            # train the model
            train_params = {
                **config
            }
            model.train(**train_params)

            # export the model
            model.export()
            
            # test the model
            new_path: dict = {
                'val': 'images/test'
            }
            edit_data(data, new_path)
            model = YOLO(f'{save}/{run_name}_{version}/{version}/weights/best.pt')
            metrics = model.val(
                save_json=True,
                batch=config['batch'],
                imgsz=config['imgsz'],
                workers=config['workers'],
                plots=True,
                project=config['project'],
                name='test',
                iou=0.1,
            )
            
            # save full metrics to file
            with open(f'{save}/{run_name}_{version}/test/metrics.txt', 'w') as file:
                file.write(str(metrics))
            
            # remove existing symlinks
            if (save / symlink_temp).exists():
                shutil.rmtree(save / symlink_temp)
    
    except Exception as e:
        # remove existing symlinks
        if (save / symlink_temp).exists():
            shutil.rmtree(save / symlink_temp)
            
        print(f'An error occured: {e}')

if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrained', type=Path,
                    help='Path to pretrained YOLO weights')
    ap.add_argument('--source-images', type=Path, required=True,
                    help='Path to training and validation dataset')
    ap.add_argument('--target-images', type=Path, required=True,
                    help='Path to training and validation dataset')
    ap.add_argument('--generated-images', type=Path, required=False, default=None,
                    help='Path to generated images')
    ap.add_argument('--save', type=Path, required=True,
                    help='Path to save weights and results.')
    ap.add_argument('--image-size', type=int, default=640,
                    help='Image size for training.')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model.')
    ap.add_argument('--batch-size', type=int, default=16,
                    help='Batch size per epoch.')
    ap.add_argument('--labels', nargs='+', default=['0'],
                    help='Labels used for training')
    ap.add_argument('--run-name', type=str, default='temp',
                    help='Name of the run.')
    ap.add_argument('--symlink-temp', type=str, default='temp',
                    help='Name of the run.')
    ap.add_argument('--model_type', type=str, default='yolo',
                    help='Type of model to train.')
    args = ap.parse_args()
    
    # set path to directory of script
    abspath = os.path.dirname(__file__)
    dname = os.path.join(os.path.dirname(abspath), 'models')
    print(f'CWD: {dname}')
    
    # defaults
    config = Path(f'{dname}/yolo_config.yaml')
    data = Path(f'{dname}/yolo_data.yaml')
    print(f'Config file: {config}')
    print(f'Data file: {data}')
    print(f'Pretrained model: {args.pretrained}')
    
    main(args.pretrained, args.source_images, args.target_images, args.generated_images, \
         config, data, args.save, args.labels, args.image_size, args.epochs, args.batch_size, \
             args.run_name, args.symlink_temp, args.model_type)