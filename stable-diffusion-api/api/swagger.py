from flask import Flask
from flasgger import APISpec, Swagger
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from marshmallow.exceptions import ValidationError

from schemas import Txt2ImgInput


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
        definitions=[('Txt2ImgInput', Txt2ImgInput)],
    )
    config = {
        "headers": [
        ],
        "specs": [
            {
                "endpoint": 'swagger',
                "route": '/swagger.json',
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
            }
        ],
        "static_url_path": "/flasgger_static",
        # "static_folder": "static",  # must be set by user
        "swagger_ui": True,
        "specs_route": "/swagger/",
    }
    Swagger(app, template=template, config=config)

    app.register_error_handler(ValidationError, handle_validation_error)


def handle_validation_error(exc):
    return {'error': 'invalid request', 'message': exc.messages}, 400