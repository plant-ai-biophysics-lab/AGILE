import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.util import visualize_attention_grid

class ImageLogger(Callback):
    def __init__(self, epoch_frequency=1, generate_images=False, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.epoch_freq = epoch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.train_dataloader = None  # To store the DataLoader

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            # if attn map skip
            if k == "attn_maps":
                continue
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()  # Move tensor to CPU before converting to NumPy
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train", save_dir="image_log"):
        check_idx = pl_module.global_step  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx, batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # add gaussian map to kwargs
                self.log_images_kwargs["gaussian_map"] = batch['attn_map']['object'].squeeze(0)
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                
                if k == "attn_maps":
                    # save attn maps here
                    attn_map_save_dir = os.path.join(pl_module.logger.save_dir, save_dir, "image_log", split)
                    os.makedirs(attn_map_save_dir, exist_ok=True)
                    attn_map_filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, pl_module.global_step, pl_module.current_epoch, batch_idx)
                    attn_map_save_dir = os.path.join(attn_map_save_dir, attn_map_filename)
                    
                    # get image from k
                    for k in images:
                        if "samples" in k:
                            rgb_img = images[k]
                    
                    images_output = images[k]
                    agg_maps, column_titles, row_titles = images_output['agg_maps'], images_output['column_titles'], images_output['row_titles']
                    visualize_attention_grid(agg_maps, rgb_img, column_titles, row_titles, attn_map_save_dir)
                    
                else:
                    
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(os.path.join(pl_module.logger.save_dir, save_dir), split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx, batch_idx):
        # return batch_idx == 0 and check_idx % self.epoch_freq == 0
        return check_idx % 2000 == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    # @rank_zero_only
    # def log_final_images(self, pl_module):
    #     if self.train_dataloader is None:
    #         return
        
    #     save_dir = os.path.join(pl_module.logger.save_dir, "final")
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     pl_module.eval()  # Ensure the model is in eval mode

    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(self.train_dataloader):
    #             images = pl_module.log_images(batch, split="final", **self.log_images_kwargs)
                
    #             for k in images:
    #                 if "samples" not in k:
    #                     continue  # Skip keys that do not contain "samples"
    #                 N = min(images[k].shape[0], self.max_images)
    #                 images[k] = images[k][:N]
    #                 if isinstance(images[k], torch.Tensor):
    #                     images[k] = images[k].detach().cpu()
    #                     if self.clamp:
    #                         images[k] = torch.clamp(images[k], -1., 1.)
                
    #             self.log_local(save_dir, "final", images, pl_module.global_step, pl_module.current_epoch, batch_idx)

    #     pl_module.train()  # Restore the model to train mode

    # def on_train_end(self, trainer, pl_module):
    #     if not self.disabled:
    #         self.log_final_images(pl_module)
