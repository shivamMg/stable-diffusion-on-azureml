import logging
from flask import Flask

from loader import ModelLoader
from txt2img import main


MODELS_NOT_LOADED_ERR = {'error': 'models not loaded yet'}

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
ModelLoader.logger = app.logger


@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    if not ModelLoader().loaded():
        return MODELS_NOT_LOADED_ERR, 503
    result = main()
    return result


@app.route('/api/health', methods=['GET'])
def health():
    if not ModelLoader().loaded():
        return MODELS_NOT_LOADED_ERR, 503
    return {'health': 'ok'}
