import logging
import os
from functools import wraps
from threading import Event, Thread

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from omegaconf import OmegaConf
from pattern_singleton import Singleton
from transformers import AutoFeatureExtractor

from ldm.util import instantiate_from_config


SAFETY_MODEL_ID = 'CompVis/stable-diffusion-safety-checker'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STABLE_DIFFUSION_DIR = os.path.join(BASE_DIR, 'stable-diffusion')
CONFIG_PATH = os.path.join(STABLE_DIFFUSION_DIR, 'configs/stable-diffusion/v1-inference.yaml')
MODEL_DIR = os.getenv('MODEL_DIR')
if MODEL_DIR:
    CKPT_PATH = os.path.join(MODEL_DIR, 'stable-diffusion-v1.ckpt')
else:
    CKPT_PATH = os.path.join(STABLE_DIFFUSION_DIR, 'models/ldm/stable-diffusion-v1/model.ckpt')


def load_models(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ModelLoader().loaded():
            return {'error': 'models not loaded yet'}, 503
        return f(*args, **kwargs)
    return decorated


class ModelLoader(metaclass=Singleton):
    logger: logging.Logger = None

    def __init__(self):
        self._loaded = Event()
        t = Thread(target=self._load_models)
        t.daemon = True
        t.start()

    def loaded(self):
        return self._loaded.is_set()

    def _load_models(self):
        self.logger.info('Loading models')
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(SAFETY_MODEL_ID)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(SAFETY_MODEL_ID)
        self.stable_diffusion = self._load_stable_diffusion_model()
        self.logger.info('Models loaded')
        self._loaded.set()

    def _load_stable_diffusion_model(self):
        pl_sd = torch.load(CKPT_PATH, map_location='cpu')
        sd = pl_sd['state_dict']
        config = OmegaConf.load(CONFIG_PATH)
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.cuda().half()  # TODO: half if not enough vram
        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        return model
