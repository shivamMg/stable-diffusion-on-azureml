from flasgger import Schema, fields
from marshmallow import validate


VALID_IMAGE_SIZES = list(range(64, 2048+1, 64))


class Txt2ImgSchema(Schema):
    prompt = fields.Str(required=True)
    seed = fields.Int()
    sampler_name = fields.Str(validate=validate.OneOf(['PLMS', 'DDIM']))
    n_samples = fields.Int()
    n_iter = fields.Int()
    latent_channels = fields.Int(attribute='C')
    height = fields.Int(validate=validate.OneOf(VALID_IMAGE_SIZES))
    width = fields.Int(validate=validate.OneOf(VALID_IMAGE_SIZES))
    downsampling_factor = fields.Int(attribute='f')
    precision = fields.Str(validate=validate.OneOf(['autocast', 'full']))
    scale = fields.Float()
    ddim_steps = fields.Int()
    ddim_eta = fields.Float()
    fixed_code = fields.Bool()
    check_safety = fields.Bool()
