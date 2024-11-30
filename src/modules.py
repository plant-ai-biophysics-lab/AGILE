import torch
import os
import math
import wandb
import time
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from tqdm import tqdm
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.checkpoint import checkpoint
    # SamProcessor, SamModel, pipeline

from src.util import default, exists, extract_into_tensor, noise_like, zero_module

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = False
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")
    
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
        
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)
    
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=10,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]][0]
                if ctmp is None:
                    ctmp = conditioning[list(conditioning.keys())[1]][0]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        # decoded synthetic image
        # x_T = kwargs['input_y'] if 'input_y' in kwargs else None
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=10,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img], 'attn_maps': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold, **kwargs)
            img, pred_x0, attn_maps = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                intermediates['attn_maps'].append({f"timestep_{index}": attn_maps})

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, **kwargs):
        b, *_, device = *x.shape, x.device
        
        # # check if `control_attentions` is True in kwargs
        # if 'control_attentions' in kwargs:
        #     control_attentions = kwargs['control_attentions']
        # else:
        #     control_attentions = False
            
        # # check if gaussian_map is in kwargs
        # if 'gaussian_map' in kwargs:
        #     gaussian_map = kwargs['gaussian_map']
        # else:
        #     gaussian_map = None

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        if c[k][0] is None:
                            c_in[k] = None
                        else:
                            c_in[k] = []
                            for i in range(len(c[k])):
                                # Apply retain_grad() before concatenation for each tensor
                                if c[k][i].requires_grad:
                                    c[k][i].retain_grad()
                                if unconditional_conditioning[k][i].requires_grad:
                                    unconditional_conditioning[k][i].retain_grad()

                                # Concatenate the tensors and store in c_in[k]
                                concatenated_tensor = torch.cat([unconditional_conditioning[k][i], c[k][i]])
                                c_in[k].append(concatenated_tensor)

                            # Only check and set requires_grad if k is 'c_crossattn'
                            if k == 'c_crossattn':
                                for i in range(len(c_in[k])):
                                    # Retain gradients for debugging purposes
                                    if c_in[k][i].requires_grad:
                                        c_in[k][i].retain_grad()

                                    # Only re-enable requires_grad if c[k][i] originally required gradients
                                    if c[k][i].requires_grad:
                                        c_in[k][i].requires_grad_(True)
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in, save_attention=True).chunk(2) 
            model_uncond_t, attn_maps = self.model.apply_model(x_in, t_in, c_in, save_attention=True)
                                                            #    control_attentions=control_attentions, gaussian_map=gaussian_map, attn_weights=self.model.model.a)
            model_uncond, model_t = model_uncond_t.chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0, attn_maps

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    
class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out
    
class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def apply_attention_edit(self, sim_target, gaussian_map, beta1=1.0, beta2=1.0, attn_weights=None, gamma=1.0):
        # Extract sizes and precompute constants
        target_size = gaussian_map.shape[-1]
        map_size = int(math.sqrt(sim_target.shape[1]))
        num_heads, num_tokens = sim_target.shape[0], sim_target.shape[2]

        # Reshape sim_target to [num_heads, num_tokens, map_size, map_size] directly
        sim_target = sim_target.view(num_heads, map_size, map_size, num_tokens).permute(0, 3, 1, 2)

        # Upsample sim_target only if necessary
        if map_size != target_size:
            sim_target = F.interpolate(sim_target, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Split token 1 and the rest
        sim_token_1 = sim_target[:, 1:2, :, :]  # Keep as a single-channel tensor
        sim_tokens_rest = sim_target[:, 2:, :, :]

        # Compute statistics for token 1 in-place
        sim_1_mean = sim_token_1.mean()
        sim_1_std = sim_token_1.std()
        gaussian_map_norm = gaussian_map * (sim_1_std * beta1) + sim_1_mean
        sim_token_1.mul_(1 - gamma).add_(gamma * gaussian_map_norm).clamp_(0.0, 1.0)

        # Compute optimized Gaussian blending for tokens 2 onwards
        selected_mean = sim_tokens_rest.mean()
        selected_std = sim_tokens_rest.std()
        gaussian_map_exp_norm = gaussian_map * (selected_std * beta2) + selected_mean
        sim_tokens_rest.mul_(1 - gamma).add_(gamma * gaussian_map_exp_norm).clamp_(0.0, 1.0)

        # Reapply attention weights if provided
        if attn_weights is not None:
            sim_tokens_rest.mul_(attn_weights.view(1, -1, 1, 1))

        # Combine token 1 and the rest
        sim_target[:, 1:2, :, :] = sim_token_1
        sim_target[:, 2:, :, :] = sim_tokens_rest

        # Downsample to original size if needed
        if target_size != map_size:
            sim_target = F.interpolate(sim_target, size=(map_size, map_size), mode='bilinear', align_corners=False)

        # Reshape back to [num_heads, map_size * map_size, num_tokens]
        return sim_target.permute(0, 2, 3, 1).reshape(num_heads, -1, num_tokens)

    def forward(self, x, context=None, mask=None, return_attn_weights=False, optimize=False, control_attentions=False, gaussian_map=None, attn_weights=None, beta1=1.0, beta2=0.1):
        b, n, _ = x.shape
        h = self.heads

        if optimize:
            with torch.enable_grad():  # Enable gradient computation for optimization
                
                # Prepare queries, keys, values
                q = self.to_q(x)
                context = default(context, x)
                k = self.to_k(context)
                v = self.to_v(context)

                q, k, v = map(
                    lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                    (q, k, v),
                )
                
                sim = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )
                if control_attentions:
                    sim = self.apply_attention_edit(
                        sim, gaussian_map, beta1=beta1, beta2=beta2, attn_weights=attn_weights
                    )
        else:
            # Prepare queries, keys, values
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (q, k, v),
            )
            
            sim = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
            if control_attentions:
                sim = self.apply_attention_edit(
                    sim, gaussian_map, beta1=beta1, beta2=beta2, attn_weights=attn_weights
                )

        # Reshape back to original dimensions
        out = (
            sim.unsqueeze(0)
            .reshape(b, h, sim.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, sim.shape[1], h * self.dim_head)
        )
        out = self.to_out(out)

        if return_attn_weights:
            return out, sim
        else:
            return out

class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None, return_attn_weights=False, optimize=False, control_attentions=False, gaussian_map=None, attn_weights=None, beta1=1.0, beta2=0.1):
        # Reshape input from (B, C, H, W) to (B, H*W, C)
        b, c, h, w = x.shape

        if optimize:
            with torch.enable_grad():  # Enable gradient computation for optimization
                
                x = rearrange(x, 'b c h w -> b (h w) c')

                h_heads = self.heads

                # Prepare queries, keys, and values
                q = self.to_q(x)
                context = default(context, x)
                k = self.to_k(context)
                v = self.to_v(context)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h_heads), (q, k, v))
                
                sim = xformers.ops.memory_efficient_attention(
                    q, k, v, attn_bias=None, op=self.attention_op
                )
                if control_attentions:
                    sim = self.apply_attention_edit(
                        sim, gaussian_map, beta1=beta1, beta2=beta2, attn_weights=attn_weights
                    )
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

            h_heads = self.heads

            # Prepare queries, keys, and values
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h_heads), (q, k, v))
            
            sim = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
            if control_attentions:
                sim = self.apply_attention_edit(
                    sim, gaussian_map, beta1=beta1, beta2=beta2, attn_weights=attn_weights
                )

        # Reshape attention output back to the original format
        sim = rearrange(sim, '(b h) n d -> b n (h d)', h=h_heads)
        out = self.to_out(sim)

        # Reshape output back to (B, C, H, W)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)

        if return_attn_weights:
            return out, sim
        else:
            return out
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self._ATTN_PRECISION = "fp32"
        self.use_checkpoint = False  # Flag to enable/disable checkpointing
        
    def apply_attention_edit(self, sim_target, gaussian_map, beta1=1.0, beta2=1.0, attn_weights=None, gamma=1.0):
        # Extract sizes and precompute constants
        target_size = gaussian_map.shape[-1]
        map_size = int(math.sqrt(sim_target.shape[1]))
        num_heads, num_tokens = sim_target.shape[0], sim_target.shape[2]

        # Reshape sim_target to [num_heads, num_tokens, map_size, map_size] directly
        sim_target = sim_target.view(num_heads, map_size, map_size, num_tokens).permute(0, 3, 1, 2)

        # Upsample sim_target only if necessary
        if map_size != target_size:
            sim_target = F.interpolate(sim_target, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Split token 1 and the rest
        sim_token_1 = sim_target[:, 1:2, :, :]  # Keep as a single-channel tensor
        sim_tokens_rest = sim_target[:, 2:, :, :]

        # Compute statistics for token 1 in-place
        sim_1_mean = sim_token_1.mean()
        sim_1_std = sim_token_1.std()
        gaussian_map_norm = gaussian_map * (sim_1_std * beta1) + sim_1_mean
        sim_token_1.mul_(1 - gamma).add_(gamma * gaussian_map_norm).clamp_(0.0, 1.0)

        # Compute optimized Gaussian blending for every 5th token
        indices = torch.arange(sim_tokens_rest.shape[1]) % 2 == 0  # Select every nth token
        selected_tokens = sim_tokens_rest[:, indices, :, :]  # Filter out every nth token
        selected_mean = selected_tokens.mean()
        selected_std = selected_tokens.std()
        gaussian_map_exp_norm = gaussian_map * (selected_std * beta2) + selected_mean
        selected_tokens = selected_tokens * (1 - gamma) + gamma * gaussian_map_exp_norm
        selected_tokens = selected_tokens.clamp(0.0, 1.0)

        # Update only every 5th token in sim_tokens_rest
        sim_tokens_rest[:, indices, :, :] = selected_tokens

        # Reapply attention weights if provided
        if attn_weights is not None:
            sim_tokens_rest.mul_(attn_weights.view(1, -1, 1, 1))

        # Combine token 1 and the rest
        sim_target[:, 1:2, :, :] = sim_token_1
        sim_target[:, 2:, :, :] = sim_tokens_rest

        # Downsample to original size if needed
        if target_size != map_size:
            sim_target = F.interpolate(sim_target, size=(map_size, map_size), mode='bilinear', align_corners=False)

        # Reshape back to [num_heads, map_size * map_size, num_tokens]
        return sim_target.permute(0, 2, 3, 1).reshape(num_heads, -1, num_tokens)

    def attention(self, q, k, v, control_attentions=False, gaussian_map=None, mask=None, attn_weights=None, beta1=1.0, beta2=0.1):
        is_fp32 = self._ATTN_PRECISION == "fp32"

        with torch.autocast(enabled=not is_fp32, device_type='cuda'):
            if is_fp32:
                q, k = q.float(), k.float()

            # Compute scaled dot-product attention
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            sim = sim.softmax(dim=-1)

            # Apply attention edits if necessary
            if control_attentions:
                sim = self.apply_attention_edit(sim, gaussian_map, attn_weights=attn_weights, beta1=beta1, beta2=beta2)

        # Apply mask if provided
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)

        # Compute attention output
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        
        return out, sim

    def forward(self, x, context=None, mask=None, return_attn_weights=False, optimize=False, control_attentions=False, gaussian_map=None, attn_weights=None, beta1=1.0, beta2=0.1):
        
        if optimize:
            with torch.enable_grad():
                h = self.heads
                q = self.to_q(x)
                context = default(context, x)
                k = self.to_k(context)
                v = self.to_v(context)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

                def checkpointed_attention(q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2):
                    return self.attention(q, k, v, control_attentions, gaussian_map, mask, attn_weights, beta1, beta2)

                if self.use_checkpoint:
                    out, sim = checkpoint(checkpointed_attention, (q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2), (), self.use_checkpoint)
                else:
                    out, sim = checkpointed_attention(q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2)
                
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                
                out = self.to_out(out)
        else:
            
            h = self.heads
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            def checkpointed_attention(q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2):
                    return self.attention(q, k, v, control_attentions, gaussian_map, mask, attn_weights, beta1, beta2)

            if self.use_checkpoint:
                out, sim = checkpoint(checkpointed_attention, (q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2), (), self.use_checkpoint)
            else:
                out, sim = checkpointed_attention(q, k, v, control_attentions, gaussian_map, attn_weights, beta1, beta2)
            
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            
            out = self.to_out(out)
        
        if return_attn_weights:
            return out, sim
        else:
            return out

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
     
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = False

    def forward(self, x, context=None, optimize=False, layer=None, control_attentions=False, gaussian_map=None, attn_weights=None, beta1=1.0, beta2=0.1):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        # print('transformer input requires grad:', x.requires_grad, context.requires_grad)
        return checkpoint(self._forward, x, context, optimize, layer, control_attentions, gaussian_map, attn_weights, beta1, beta2)

    def _forward(self, x, context=None, optimize=False, layer=None, control_attentions=False, gaussian_map=None, attn_weights=None, beta1=1.0, beta2=0.1):
        if optimize:
            with torch.enable_grad() and torch.autograd.graph.save_on_cpu():
                attn1_output = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, return_attn_weights=False, optimize=optimize)
                # self.attn_maps['attn1'] = attn1_weights  # Store the attention weights
                x = attn1_output + x  # Update x with the output of attn1

                attn2_output, attn2_weights = self.attn2(
                    self.norm2(x), context=context, return_attn_weights=True, optimize=optimize, control_attentions=control_attentions, 
                    gaussian_map=gaussian_map, attn_weights=attn_weights, beta1=beta1, beta2=beta2
                )
                attn_maps = {'attn2': attn2_weights}
                x = attn2_output + x  # Update x with the output of attn2

                x = self.ff(self.norm3(x)) + x
            
        else:
            attn1_output = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, return_attn_weights=False, optimize=optimize)
            # self.attn_maps['attn1'] = attn1_weights  # Store the attention weights
            x = attn1_output + x  # Update x with the output of attn1

            attn2_output, attn2_weights = self.attn2(
                self.norm2(x), context=context, return_attn_weights=True, optimize=optimize, control_attentions=control_attentions,
                gaussian_map=gaussian_map, attn_weights=attn_weights, beta1=beta1, beta2=beta2
            )
            attn_maps = {'attn2': attn2_weights}
            x = attn2_output + x  # Update x with the output of attn2

            x = self.ff(self.norm3(x)) + x
        
        if layer is not None:
            return x, attn_maps
        else:
            return x
    
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        
        # check if attention layer to save
        attn_maps = []
        if 'layer' in kwargs:
            layer = kwargs['layer']
        else:
            layer = None
            
        # check if optimizing
        if 'optimizing' in kwargs:
            optimize = kwargs['optimizing']
        else:
            optimize = False
            
        # check if control attentions
        if 'control_attentions' in kwargs:
            control_attentions = kwargs['control_attentions']
        else:
            control_attentions = False
            
        # check if gaussian map
        if 'gaussian_map' in kwargs:
            gaussian_map = kwargs['gaussian_map']
        else:
            gaussian_map = None
            
        # check if attn_weights
        if 'attn_weights' in kwargs:
            attn_weights = kwargs['attn_weights']
        else:
            attn_weights = None
            
        # get betas
        if 'beta1' in kwargs:
            beta1 = kwargs['beta1']
        else:
            beta1 = 1.0
            
        if 'beta2' in kwargs:
            beta2 = kwargs['beta2']
        else:
            beta2 = 0.1
        
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
            
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        
        if not self.use_linear:
            x = self.proj_in(x)
            
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        
        if self.use_linear:
            x = self.proj_in(x)
        
        for i, block in enumerate(self.transformer_blocks):
            # if context[i] has shape (b, c, h, w), we need to reshape to (b, h*w, c)
            if len(context[i].shape) == 4:
                context[i] = rearrange(context[i], 'b c h w -> b (h w) c').contiguous()

            if layer is not None:
                x, attn_map = block(x, context=context[i], optimize=optimize, layer=layer, 
                                    control_attentions=control_attentions, gaussian_map=gaussian_map, attn_weights=attn_weights, beta1=beta1, beta2=beta2)
                attn_maps.append(attn_map)
            else:
                x = block(x, context=context[i], optimize=optimize, layer=layer)
        
        if self.use_linear:
            x = self.proj_out(x)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        
        if not self.use_linear:
            x = self.proj_out(x)
        
        if layer is not None:
            return x + x_in, attn_maps
        
        return x + x_in

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
            
def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    
def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

class DDIMSamplerWithGrad(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @staticmethod
    def parse_attn_maps(attn_maps, target_size=512):
        all_maps = []

        for _, (_, attn_map) in enumerate(attn_maps.items()):
            for attn_type, values in attn_map[0].items():
                if attn_type == 'attn2':
                    # Directly compute the mean over the batch
                    a_map = values.mean(dim=0)  # Shape: [256, 77]

                    # Ensure the spatial map size matches the expected 16x16 grid
                    num_spatial_elements = a_map.shape[0]  # 256 spatial elements
                    num_tokens = a_map.shape[1]  # 77 tokens
                    map_size = int(math.sqrt(num_spatial_elements))  # 16x16 spatial size

                    if num_spatial_elements != map_size * map_size:
                        raise ValueError(f"Cannot reshape {num_spatial_elements} spatial elements into a square grid.")

                    # Reshape and interpolate in one step
                    a_map = a_map.view(map_size, map_size, num_tokens).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 77, 16, 16]
                    
                    # Interpolate directly
                    a_map = nn.functional.interpolate(a_map, size=(target_size, target_size), mode='bilinear', align_corners=False)

                    # Squeeze and reorder back to [target_size, target_size, num_tokens]
                    all_maps.append(a_map.squeeze(0))  # Shape: [77, target_size, target_size]

        # Stack all attention maps and take the mean across layers in one go
        all_maps_in_layer = torch.stack(all_maps, dim=0).mean(dim=0)  # Shape: [77, target_size, target_size]

        # Normalize in-place to avoid extra memory allocation
        all_maps_in_layer = (all_maps_in_layer - all_maps_in_layer.min()) / (
            all_maps_in_layer.max() - all_maps_in_layer.min())

        # Permute back to [target_size, target_size, num_tokens]
        return all_maps_in_layer.permute(1, 2, 0)  # Shape: [target_size, target_size, num_tokens]
    
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=10,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]][0]
                if ctmp is None:
                    ctmp = conditioning[list(conditioning.keys())[1]][0]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    **kwargs
                                                    )
        return samples, intermediates

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=10,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        img.requires_grad = True

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img], 'attn_maps': []}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        #### ATTENTION GUIDANCE ####
        control_attentions = kwargs['control_attentions'] if 'control_attentions' in kwargs else False
        if control_attentions:
            
            # Add a to kwargs
            # target_map = kwargs['gaussian_map']  # Assume Gaussian map is [512, 512]
            # batch_idx = kwargs['batch_idx']
            # logs_dir = kwargs['logs_dir']
            # source_img = kwargs['source_img']
            betas = kwargs['betas']
            # opt_steps = 1
            
            # Prefill the grid for normal images and prompt maps
            # num_timesteps_to_display = 6  # For Timestep 50, 40, 30, 20, 10, 0
            # grid_images = [[None for _ in range(num_timesteps_to_display + 1)] for _ in range(opt_steps)]  # +1 for source_img
            
            # Prefill the grid for prompt maps with timesteps in columns and rows are tokens [1, 5, 10, 30, 50, 70]
            # tokens_to_display = [1, 5, 10, 30, 50, 70]
            # prompt_map_grid = [[None for _ in range(num_timesteps_to_display)] for _ in range(len(tokens_to_display))]  # +1 for target_map
            
            # Timesteps to display (every 10th step, starting from 50)
            # timesteps_to_display = [49, 40, 30, 20, 10, 0]
            
            # for opt_step in range(opt_steps):
            for i, step in enumerate(iterator):

                index = total_steps - i - 1

                if 'control_attentions' in kwargs:
                    kwargs['control_attentions'] = True
                    if index >= 40:
                        beta1 = betas[0][0]
                        beta2 = betas[0][1]
                        kwargs['beta1'] = beta1
                        kwargs['beta2'] = beta2
                    elif index >= 15:
                        beta1 = betas[1][0]
                        beta2 = betas[1][1]
                        kwargs['beta1'] = beta1
                        kwargs['beta2'] = beta2
                    else:
                        beta1 = betas[2][0]
                        beta2 = betas[2][1]
                        kwargs['beta1'] = beta1
                        kwargs['beta2'] = beta2

                ts = torch.full((b,), step, device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)
                    img = img_orig * mask + (1. - mask) * img

                if ucg_schedule is not None:
                    assert len(ucg_schedule) == len(time_range)
                    unconditional_guidance_scale = ucg_schedule[i]

                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold, **kwargs)
                img, pred_x0, attn_maps = outs
                    
                # ### NOTE FOR DEBUGGING NOTE ###
                #     # Check if `index` is in `timesteps_to_display`
                #     if index in timesteps_to_display:
                #         timestep_index = timesteps_to_display.index(index)
                #         img_decoded = self.model.decode_first_stage(img)
                #         grid_images[opt_step][timestep_index] = img_decoded  # Store in prefilled grid for selected timesteps
                        
                #         # Save the corresponding `prompt_map` in the prompt_map_grid if in last opt_step
                #         if opt_step == opt_steps - 1:
                            
                #             # get tokens to display from avg_attn_maps
                #             display_avg_attn_maps = self.parse_attn_maps(attn_maps)
                #             tokens_to_display_avg = display_avg_attn_maps[:, :, tokens_to_display]
                #             for i, _ in enumerate(tokens_to_display):
                #                 token_colored = self.apply_viridis_colormap(tokens_to_display_avg[:, :, i])
                #                 prompt_map_grid[i][timestep_index] = token_colored
                    
                # # Add source_img and target_map at the end of each row
                # grid_images[opt_step][-1] = source_img.squeeze(0).permute(2, 0, 1)  # Add source image at the end of the row in the normal grid
                # ### NOTE FOR DEBUGGING NOTE ###
                
                # ### NOTE FOR DEBUGGING NOTE ###
                # # Reset img
                # if x_T is None:
                #     img = torch.randn(shape, device=device)
                # else:
                #     img = x_T
                # kwargs['control_attentions'] = True
                # print('Setting control attentions back to True')
                # iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
                # ### NOTE FOR DEBUGGING NOTE ###

            #  ### NOTE FOR DEBUGGING NOTE ###
            # # Generate the image grid from the prefilled array for both normal images and prompt maps
            # opt_step_titles = [f'Step {i + 1}' for i in range(opt_steps)]  # Row titles
            
            #  # Save normal image grid
            # rgb_img_dir = os.path.join(logs_dir, 'rgb_images')
            # self.create_image_grid_prefilled(grid_images, opt_steps, num_timesteps_to_display + 1, rgb_img_dir, timesteps_to_display + ["Source"], opt_step_titles, batch_idx=batch_idx, target_map=target_map, final_img=None)
            
            # # Save prompt map grid
            # attn_map_dir = os.path.join(logs_dir, 'attn_maps')
            # self.create_prompt_map_grid_prefilled(prompt_map_grid, len(tokens_to_display), num_timesteps_to_display, attn_map_dir, timesteps_to_display, tokens_to_display, batch_idx)
            #  ### NOTE FOR DEBUGGING NOTE ###
            
        else:
            kwargs['optimizing'] = True
            for i, step in enumerate(iterator):
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)

                    if mask is not None:
                        assert x0 is not None
                        img_orig = self.model.q_sample(x0, ts)
                        img = img_orig * mask + (1. - mask) * img

                    if ucg_schedule is not None:
                        assert len(ucg_schedule) == len(time_range)
                        unconditional_guidance_scale = ucg_schedule[i]

                    outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                            quantize_denoised=quantize_denoised, temperature=temperature,
                                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                                            corrector_kwargs=corrector_kwargs,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning,
                                            dynamic_threshold=dynamic_threshold, **kwargs)
                    img, pred_x0, attn_maps = outs

                    if callback: callback(i)
                    if img_callback: img_callback(pred_x0, i)

                    if index % log_every_t == 0 or index == total_steps - 1:
                        intermediates['x_inter'].append(img)
                        intermediates['pred_x0'].append(pred_x0)
                        intermediates['attn_maps'].append({f"timestep_{index}": attn_maps})

        return img, intermediates

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None, use_checkpoint=True, **kwargs):
        b, *_, device = *x.shape, x.device
        
        # check if `control_attentions` is True in kwargs
        if 'control_attentions' in kwargs:
            control_attentions = kwargs['control_attentions']
        else:
            control_attentions = False
            
        # check if gaussian_map is in kwargs
        if 'gaussian_map' in kwargs:
            gaussian_map = kwargs['gaussian_map']
        else:
            gaussian_map = None
            
        if 'a_vector' in kwargs:
            a_vector = kwargs['a_vector']
        else:
            a_vector = None
            
        if 'optimizing' in kwargs:
            optimizing = kwargs['optimizing']
        else:
            optimizing = False
            
        if 'beta1' in kwargs:
            beta1 = kwargs['beta1']
        else:
            beta1 = 1.0
            
        if 'beta2' in kwargs:
            beta2 = kwargs['beta2']
        else:
            beta2 = 0.1

        def forward_model(x_in, t_in, c_in):
            if control_attentions:
                return self.model.apply_model(x_in, t_in, c_in, 
                            save_attention=True, optimizing=False, control_attentions=True, gaussian_map=gaussian_map, attn_weights=a_vector, beta1=beta1, beta2=beta2
                        )
            return self.model.apply_model(x_in, t_in, c_in, save_attention=True, optimizing=optimizing)

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # Use checkpointing on the forward model
            if use_checkpoint:
                # check if inputs need grads
                # print('input gradients:', x.requires_grad, t.requires_grad, c.requires_grad)
                model_output, attn_maps = checkpoint(forward_model, x, t, c, control_attentions)
            else:
                model_output, attn_maps = forward_model(x, t, c, control_attentions)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        if c[k][0] is None:
                            c_in[k] = None
                            
                        # check if already concatentated
                        elif len(c[k]) == 2 and c[k][0].shape[0] == b:
                            c_in[k] = c[k]
                        else:
                            c_in[k] = []
                            for i in range(len(c[k])):
                                if k == 'c_crossattn':
                                    if c[k][i].requires_grad:
                                        c[k][i].retain_grad()
                                concatenated_tensor = torch.cat([unconditional_conditioning[k][i], c[k][i]])
                                concatenated_tensor.requires_grad_()
                                c_in[k].append(concatenated_tensor)
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])

            # Use checkpointing for concatenated tensors
            if use_checkpoint:
                # print('input gradients:', x_in.requires_grad, t_in.requires_grad, c_in['c_crossattn'][0].requires_grad)
                model_uncond_t, attn_maps = checkpoint(forward_model, x_in, t_in, c_in)
            else:
                model_uncond_t, attn_maps = forward_model(x_in, t_in, c_in)
            
            model_uncond, model_t = model_uncond_t.chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0, attn_maps
    
    @staticmethod
    def apply_viridis_colormap(attn_map):
        """Apply the VIRIDIS colormap to a 2D attention map."""
        
        # Clone the tensor if you need to preserve the computation graph
        attn_map_for_colormap = attn_map.clone().detach().cpu().squeeze()  # Detach and move to CPU for NumPy
        
        attn_map_np = attn_map_for_colormap.numpy()  # Convert to NumPy array
        viridis = cm.get_cmap('viridis')
        attn_map_colored = viridis(attn_map_np)[:, :, :3]  # Apply colormap and drop alpha channel
        attn_map_colored = torch.tensor(attn_map_colored).permute(2, 0, 1)  # Ensure it's in [3, H, W] format
        
        return attn_map_colored
        
    @staticmethod
    def create_image_grid_prefilled(images_grid, rows, cols, logs_dir, timesteps, opt_steps, batch_idx, target_map=None, final_img=None):

        # Determine the device (if all tensors should be on the same GPU or CPU)
        device = images_grid[0][0].device if images_grid[0][0] is not None else 'cpu'

        # Create a flat list of images from the prefilled grid for stacking
        processed_images = []
        for row in images_grid:
            for img in row:
                if img is not None:
                    img = img.to(device)
                    # Remove batch dimension if present
                    if img.dim() == 4 and img.shape[0] == 1:  
                        img = img.squeeze(0)  # Now img is [3, 512, 512] or [1, 512, 512]
                    # If the image is grayscale (1 channel), convert it to 3 channels
                    if img.shape[0] == 1:  
                        img = img.repeat(3, 1, 1)  # Convert grayscale to RGB
                    processed_images.append(img)
                else:
                    processed_images.append(torch.zeros(3, 512, 512).to(device))  # Placeholder for missing images

        # Stack the images into a single tensor
        grid_img = vutils.make_grid(torch.stack(processed_images), nrow=cols, padding=2, normalize=True)

        # Create the logs directory if it doesn't exist
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Create the plot for the grid
        plt.figure(figsize=(20, 10))
        plt.imshow(grid_img.permute(1, 2, 0).cpu())  # Move to CPU for displaying and change dimensions for matplotlib
        plt.axis('off')

        # Add column titles (timesteps) and row titles (opt_steps)
        for col in range(cols):
            plt.text(col * grid_img.shape[2] // cols + grid_img.shape[2] // (2 * cols),
                    -10, f'Timestep {timesteps[col]}', ha='center', va='bottom', fontsize=10, color='black')

        for row in range(rows):
            plt.text(-20, row * grid_img.shape[1] // rows + grid_img.shape[1] // (2 * rows),
                    f'Opt Step {opt_steps[row]}', ha='right', va='center', fontsize=10, color='black')

        # Save the grid image with row and column titles
        save_path = os.path.join(logs_dir, f"batch_idx_{batch_idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Grid image with titles saved to {save_path}")
            
    @staticmethod
    def create_prompt_map_grid_prefilled(prompt_map_grid, rows, cols, logs_dir, timesteps, tokens, batch_idx):
        # Determine the device (if all tensors should be on the same GPU or CPU)
        device = prompt_map_grid[0][0].device if prompt_map_grid[0][0] is not None else 'cpu'

        # Create a flat list of images from the prefilled grid for stacking
        processed_images = []
        for row in prompt_map_grid:
            for img in row:
                if img is not None:
                    img = img.to(device)
                    # Remove batch dimension if present
                    if img.dim() == 4 and img.shape[0] == 1:
                        img = img.squeeze(0)  # Now img is [3, 512, 512] or [1, 512, 512]
                    # If the image is grayscale (1 channel), convert it to 3 channels
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)  # Convert grayscale to RGB
                    processed_images.append(img)
                else:
                    processed_images.append(torch.zeros(3, 512, 512).to(device))  # Placeholder for missing images

        # Stack the images into a single tensor
        grid_img = vutils.make_grid(torch.stack(processed_images), nrow=cols, padding=2, normalize=True)

        # Create the logs directory if it doesn't exist
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Create the plot for the grid
        plt.figure(figsize=(20, 10))
        plt.imshow(grid_img.permute(1, 2, 0).cpu())  # Move to CPU for displaying and change dimensions for matplotlib
        plt.axis('off')

        # Add column titles (timesteps) and row titles (tokens)
        for col in range(cols):
            plt.text(col * grid_img.shape[2] // cols + grid_img.shape[2] // (2 * cols),
                    -10, f'Timestep {timesteps[col]}', ha='center', va='bottom', fontsize=10, color='black')

        for row in range(rows):
            plt.text(-20, row * grid_img.shape[1] // rows + grid_img.shape[1] // (2 * rows),
                    f'Token {tokens[row]}', ha='right', va='center', fontsize=10, color='black')

        # Save the grid image with row and column titles
        save_path = os.path.join(logs_dir, f"batch_idx_{batch_idx}_prompt_map.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Prompt map grid image with titles saved to {save_path}")