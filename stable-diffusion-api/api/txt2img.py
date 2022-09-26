#!/bin/env python

import os
import base64
import torch
from io import BytesIO
import numpy as np
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from loader import ModelLoader


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x
def check_safety(x_image):
    safety_checker_input = ModelLoader().safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = ModelLoader().safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


class Txt2ImgOptions:
    def __init__(self, prompt, seed=42, sampler_name='PLMS', n_samples=1, n_iter=1, C=4, H=512, W=512, f=8,
        precision='autocast', scale=7.5, ddim_steps=50, ddim_eta=0.0, outdir='outputs/mytxt2img-samples',
        fixed_code=False, check_safety=True):
        self.prompt = prompt
        self.seed = seed
        self.sampler_name = sampler_name
        self.n_samples = n_samples
        self.n_iter = n_iter  # https://github.com/CompVis/stable-diffusion/issues/218#issuecomment-1241654651
        self.C = C
        self.H = H
        self.W = W
        self.f = f
        self.precision = precision  # 'autocast' or 'full'
        self.scale = scale
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.fixed_code = fixed_code
        self.check_safety = check_safety
        self.outdir = outdir
 

def txt2img(opts: Txt2ImgOptions):
    seed_everything(opts.seed)

    model = ModelLoader().stable_diffusion

    SAMPLER_CLASS = {
        'PLMS': PLMSSampler,
        'DDIM': DDIMSampler,
    }
    sampler = SAMPLER_CLASS[opts.sampler_name](model)

    batch_size = opts.n_samples
    prompt = opts.prompt
    data = [batch_size * [prompt]]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    start_code = None
    if opts.fixed_code:
        start_code = torch.randn([opts.n_samples, opts.C, opts.H // opts.f, opts.W // opts.f], device=device)

    os.makedirs(opts.outdir, exist_ok=True)
    base_count = len(os.listdir(opts.outdir))

    result = {'iterations': []}
    precision_scope = autocast if opts.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in range(opts.n_iter):
                    iteration = []
                    for prompts in data:
                        uc = None
                        if opts.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)
                        shape = [opts.C, opts.H // opts.f, opts.W // opts.f]
                        samples_ddim, _ = sampler.sample(S=opts.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opts.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opts.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opts.ddim_eta,
                                                            x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        if opts.check_safety:
                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        else:
                            x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            buffer = BytesIO()
                            img.save(buffer, format='PNG')
                            b64 = base64.b64encode(buffer.getvalue())
                            # with open(os.path.join(sample_path, f"{base_count:05}.txt"), 'wb') as f:
                            #     f.write(b64)

                            iteration.append({'image': {
                                'type': 'BASE64',
                                'base64': b64.decode('utf-8'),
                            }})

                            base_count += 1
                    result['iterations'].append(iteration)
    return result


if __name__ == '__main__':
    txt2img()
