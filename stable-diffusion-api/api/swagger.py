from flask import Flask
from flasgger import APISpec, Swagger
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow.exceptions import ValidationError

from schemas import Txt2ImgSchema


def init_swagger(app: Flask):
    spec = APISpec(
        title='Stable Diffusion',
        version='0.0.0',
        openapi_version='2.0',
        plugins=[
            FlaskPlugin(),
            MarshmallowPlugin(),
        ],
    )
    template = spec.to_flasgger(
        app,
        definitions=[('Txt2Img', Txt2ImgSchema)],
    )
    Swagger(app, template=template)

    app.register_error_handler(ValidationError, handle_validation_error)


def handle_validation_error(exc):
    return {'error': 'invalid request', 'message': exc.messages}, 400