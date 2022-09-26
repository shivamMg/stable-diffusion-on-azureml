import logging

from flask import Flask, request
from flasgger import swag_from

from loader import ModelLoader, load_models
from schemas import Txt2ImgInput
from swagger import init_swagger
from txt2img import txt2img, Txt2ImgOptions


app = Flask(__name__)
app.logger.setLevel(logging.INFO)
ModelLoader.logger = app.logger
init_swagger(app)


@app.before_request
def log_request():
    app.logger.info('Incoming request: %s %s', request.method, request.path)
    if request.get_json(silent=True):
        app.logger.info('Request body: %s', request.json)


@app.after_request
def log_response(response):
    app.logger.info('Outgoing response: %s', response.status)
    return response


@app.route('/api/txt2img', methods=['POST'])
@load_models
@swag_from('./specs/txt2img.yml')
def txt2img_api():
    opts = Txt2ImgInput().load(request.json)
    app.logger.info('Serialized options: %s', opts)
    result = txt2img(Txt2ImgOptions(**opts))
    return result


@app.route('/api/health', methods=['GET'])
@load_models
def health():
    return {'health': 'ok'}
