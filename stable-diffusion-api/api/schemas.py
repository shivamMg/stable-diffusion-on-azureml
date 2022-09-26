from email.policy import default
from flasgger import Schema, fields
from marshmallow import validate


VALID_IMAGE_SIZES = list(range(64, 2048+1, 64))


class Txt2ImgInput(Schema):
    prompt = fields.Str(required=True)
    seed = fields.Int(default=42)
    sampler_name = fields.Str(default='PLMS', validate=validate.OneOf(['PLMS', 'DDIM']))
    n_samples = fields.Int(default=1)
    n_iter = fields.Int(default=1)
    latent_channels = fields.Int(default=4, attribute='C')
    height = fields.Int(default=512, validate=validate.OneOf(VALID_IMAGE_SIZES))
    width = fields.Int(default=512, validate=validate.OneOf(VALID_IMAGE_SIZES))
    downsampling_factor = fields.Int(default=8, attribute='f')
    precision = fields.Str(default='autocast', validate=validate.OneOf(['autocast', 'full']))
    scale = fields.Float(default=7.5)
    ddim_steps = fields.Int(default=50)
    ddim_eta = fields.Float(default=0.0)
    fixed_code = fields.Bool(default=False)
    check_safety = fields.Bool(default=True)
