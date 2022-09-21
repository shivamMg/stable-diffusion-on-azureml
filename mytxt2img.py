#!/bin/env python

import os
import base64
import torch
from io import BytesIO
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
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
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # TODO: half if not enough vram
    model.cuda().half()

    model.eval()
    return model

class Opt:
    pass

def main():
    opt = Opt()
    model_dir = os.getenv("MODEL_DIR", "models/ldm/stable-diffusion-v1")

    opt.seed = 1
    opt.config = 'configs/stable-diffusion/v1-inference.yaml'
    opt.ckpt = os.path.join(model_dir, "model.ckpt")
    opt.sampler_name = 'PLMS' or 'DDIM'
    opt.n_samples = 1
    opt.prompt = 'astronaut riding a horse'
    opt.C = 4
    opt.H = 256
    opt.W = 256
    opt.f = 8
    opt.precision = 'autocast' or 'full'
    opt.scale = 7.5
    opt.ddim_steps = 50
    opt.ddim_eta = 0.0
    opt.outdir = "outputs/mytxt2img-samples"
    opt.fixed_code = False
    # https://github.com/CompVis/stable-diffusion/issues/218#issuecomment-1241654651
    opt.n_iter = 1

    # begin
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    SAMPLER_CLASS = {
        'PLMS': PLMSSampler,
        'DDIM': DDIMSampler,
    }
    sampler = SAMPLER_CLASS[opt.sampler_name](model)

    batch_size = opt.n_samples
    prompt = opt.prompt
    data = [batch_size * [prompt]]

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    sample_path = opt.outdir
    base_count = len(os.listdir(opt.outdir))

    result = {'iterations': []}
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in range(opt.n_iter):
                    iteration = []
                    for prompts in data:
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

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
                                'base64': b64.decode('ascii'),
                            }})

                            base_count += 1
                    result['iterations'].append(iteration)
    return result


if __name__ == '__main__':
    main()