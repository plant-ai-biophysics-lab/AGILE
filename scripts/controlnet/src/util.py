import importlib
import torch
import math
import os
import numpy as np
import torch.nn as nn

from einops import repeat
from omegaconf import OmegaConf
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2

def initialize_weights(tensor):
    if tensor.ndimension() == 2:  # Check if the tensor is a linear layer weight
        nn.init.kaiming_normal_(tensor)
    elif tensor.ndimension() == 4:  # Check if the tensor is a conv layer weight
        nn.init.kaiming_normal_(tensor)
    else:
        nn.init.constant_(tensor, 0)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

def instantiate_cond_stage(self, config):
    if not self.cond_stage_trainable:
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            self.cond_stage_model = None
            # self.be_unconditional = True
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
    else:
        assert config != '__is_first_stage__'
        assert config != '__is_unconditional__'
        model = instantiate_from_config(config)
        self.cond_stage_model = model
        
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
    
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    
def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class PermuteTransform:
    def __call__(self, x):
        # Permute dimensions from [512, 512, 3] to [512, 3, 512]
        # Only permute if 3 channels are present
        if len(np.array(x).shape) == 2:
            return x
        else:
            return np.transpose(x, (0, 1, 2))
    
def get_attn_maps(attn_maps, num_layers = 9):
    
    # store attention type into a display grid (columns: timesteps and rows: layers)
    
    # create empty array to store attention maps (single column first of length layers)
    agg_maps = torch.zeros((num_layers+1, len(attn_maps), 512, 512))
    att_types = ['attn2']
    target_size = (512, 512)
    column_titles = []
    row_titles = ['Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 
                  'Layer 7', 'Layer 8', 'Layer 9', 'Layer 10',
                  'Layer 11', 'Average']
    
    # iterate attn map for each timestep
    for i, attn_map_step in enumerate(attn_maps):
        
        # get timestep key and value
        (timestep, attn_map_layers), = attn_map_step.items()
        column_titles.append(timestep)
        
        # loop through each layer
        all_maps_in_layer = []
        for j, (layer, attn_map) in enumerate(attn_map_layers.items()):
            
            for attn_type, values in attn_map[0].items():
                    
                # only keep attn_types
                if attn_type in att_types:
                    
                    # get the attention map
                    values = values[:, :, 1] # keep only the class token # TODO: use the first token (grape)
                    a_map = values.mean(dim=0) # take the mean
                    
                    # reshape into grid
                    map_size = int(math.sqrt(a_map.shape[-1]))
                    a_map = a_map.view(map_size, map_size)
                    
                    # assuming shape is (heads, sequence length), add batch dimension and head dimension
                    a_map = a_map.unsqueeze(0).unsqueeze(0)
                    
                    # interpolate to target size
                    a_map = nn.functional.interpolate(a_map, size=target_size, mode='bilinear', align_corners=False)
                    
                    # remove batch and head dimension
                    a_map = a_map.squeeze(0).squeeze(0)
                    
                    # add to list of maps for averaging
                    all_maps_in_layer.append(a_map)
                    
                    # store attention map in the display grid
                    agg_maps[j, i] = a_map.cpu().detach()
            
        # add avg map to display grid
        all_maps_in_layer = torch.stack(all_maps_in_layer, dim=0)
        all_maps_in_layer = all_maps_in_layer.mean(dim=(0))
        agg_maps[j+1, i] = all_maps_in_layer.cpu().detach()
    
    # normalize all maps to [0, 1]
    # agg_maps = (agg_maps - agg_maps.min()) / (agg_maps.max() - agg_maps.min())
    
    return {
        'agg_maps': agg_maps,
        'column_titles': column_titles,
        'row_titles': row_titles
    }
    
def visualize_attention_grid(agg_maps, rgb_image, column_titles, row_titles, save_path, alpha=0.8):
    """
    Visualizes aggregated attention maps in a grid layout and saves the visualization as an image.

    :param agg_maps: Tensor of shape (num_layers, num_timesteps, H, W) containing attention maps.
    :param rgb_image: RGB image tensor.
    :param column_titles: List of column titles (timesteps).
    :param row_titles: List of row titles (layer names).
    :param save_path: File path to save the visualization image.
    :param alpha: Alpha value for overlay transparency.
    """
    num_layers, num_timesteps, H, W = agg_maps.shape

    # Prepare the RGB image for overlay
    rgb_image = rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, 3)
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # Normalize to [0, 1]
    rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8 for display

    fig, axes = plt.subplots(nrows=num_layers, ncols=num_timesteps, figsize=(num_timesteps * 2, num_layers * 2))

    for i in range(num_layers):
        for j in range(num_timesteps):
            ax = axes[i, j]
            attention_map = agg_maps[i, j].cpu().detach().numpy()

            # Normalize the attention map
            attention_map -= attention_map.min()
            attention_map /= attention_map.max()

            # Resize attention map to match the size of the RGB image
            attention_resized = cv2.resize(attention_map, (rgb_image.shape[1], rgb_image.shape[0]))

            # Apply colormap to the attention map (change from JET to VIRIDIS)
            attention_colored = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Overlay attention map on the RGB image
            overlayed_image = (1 - alpha) * rgb_image + alpha * attention_colored
            overlayed_image = np.clip(overlayed_image / 255, 0, 1)  # Normalize for display

            # Display the overlayed image
            ax.imshow(overlayed_image, aspect='auto')
            ax.axis('off')

            # Set row titles (layer names)
            if j == 0:
                ax.axis('on')
                ax.set_xticks([])  # Hide x-axis ticks
                ax.set_yticks([])  # Hide y-axis ticks
                ax.set_ylabel(row_titles[i], rotation=90, size='small', labelpad=20, va='center_baseline', ha='right')

            # Set column titles (timestep values)
            if i == 0:
                ax.set_title(column_titles[j], size='small')
                
    # Add colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes.ravel().tolist(), shrink=0.95, orientation='horizontal', pad=0.05)
    cbar.set_label('Attention Intensity')

    # Save the figure as an image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')